package proxy

// Anthropic Messages 格式（/v1/messages）适配层。
//
// 只做格式转换：请求转成内部的 chatRequest，复用与 /v1/chat/completions 相同的
// 选号 → 发消息 → 轮询流程（prepareChat / runChat），再把回复包成 Anthropic 的
// message JSON 或 SSE 事件流。
//
// 限制与 OpenAI 接口相同：上游只收到最后一条用户消息的文字（和附件），
// system、tools、tool_use / tool_result 都不会传给 Sumus，模型永远不会发起工具调用。
// Claude Code 接上以后能对话，但读写文件、跑命令都用不了。

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
	"time"
)

// anthropicRequest 是 /v1/messages 请求体（只取用得到的字段）。
type anthropicRequest struct {
	Model     string             `json:"model"`
	Messages  []anthropicMessage `json:"messages"`
	System    json.RawMessage    `json:"system"`
	Stream    bool               `json:"stream"`
	MaxTokens int                `json:"max_tokens"`
	Metadata  struct {
		UserID string `json:"user_id"`
	} `json:"metadata"`
}

type anthropicMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

// anthropicBlock 覆盖 text / image / document / tool_result 四种会用到的内容块。
// tool_use、thinking 等块直接跳过。
type anthropicBlock struct {
	Type   string `json:"type"`
	Text   string `json:"text"`
	Title  string `json:"title"`
	Source *struct {
		Type      string `json:"type"` // base64 | url | text
		MediaType string `json:"media_type"`
		Data      string `json:"data"`
		URL       string `json:"url"`
	} `json:"source"`
	Content json.RawMessage `json:"content"` // tool_result 的内容：字符串或块数组
}

// handleMessages 处理 Anthropic Messages 请求，支持流式与非流式。
func (h *Handler) handleMessages(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeAnthropicError(w, http.StatusMethodNotAllowed, "仅支持 POST")
		return
	}
	start := time.Now()

	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeAnthropicError(w, http.StatusBadRequest, "读取请求体失败")
		return
	}
	var areq anthropicRequest
	if err := json.Unmarshal(body, &areq); err != nil {
		writeAnthropicError(w, http.StatusBadRequest, "JSON 解析失败: "+err.Error())
		return
	}
	if len(areq.Messages) == 0 {
		writeAnthropicError(w, http.StatusBadRequest, "messages 不能为空")
		return
	}

	req := chatRequest{
		Model:    mapAnthropicModel(areq.Model),
		Stream:   areq.Stream,
		Messages: make([]chatMessage, 0, len(areq.Messages)),
	}
	for _, m := range areq.Messages {
		req.Messages = append(req.Messages, chatMessage{Role: m.Role, Content: anthropicToOpenAIContent(m.Content)})
	}
	inputTokens := estimateAnthropicInput(areq)

	// max_tokens=1 的请求拿不到有用的回复（Claude Code 启动时用它探测额度），
	// 直接本地应答，不去上游花积分。
	if areq.MaxTokens == 1 {
		h.writeAnthropicReply(w, areq, "ok", "local-"+fmt.Sprint(start.UnixNano()), inputTokens, "max_tokens")
		return
	}

	convKey := anthropicConvKey(r, areq, req)
	job, aerr := h.prepareChat(req, convKey, start)
	if aerr != nil {
		writeAnthropicError(w, aerr.Status, aerr.Message)
		return
	}
	defer job.Release()

	if !areq.Stream {
		text, msgID, aerr := h.runChat(r.Context(), job)
		if aerr != nil {
			writeAnthropicError(w, aerr.Status, aerr.Message)
			return
		}
		h.writeAnthropicReply(w, areq, text, msgID, inputTokens, "end_turn")
		return
	}

	// 流式：先发 message_start，等上游时每 10 秒发 ping 保活
	// （上游是轮询模型，回复可能要等几十秒，不保活客户端会判超时）。
	sse := newSSE(w)
	msgID := "msg_" + fmt.Sprint(start.UnixNano())
	sse.event("message_start", map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id": msgID, "type": "message", "role": "assistant", "model": areq.Model,
			"content": []any{}, "stop_reason": nil, "stop_sequence": nil,
			"usage": map[string]any{"input_tokens": inputTokens, "output_tokens": 0},
		},
	})
	sse.event("ping", map[string]any{"type": "ping"})

	type result struct {
		text string
		err  *apiError
	}
	done := make(chan result, 1)
	go func() {
		text, _, aerr := h.runChat(r.Context(), job)
		done <- result{text, aerr}
	}()

	tick := time.NewTicker(10 * time.Second)
	defer tick.Stop()
	var res result
wait:
	for {
		select {
		case res = <-done:
			break wait
		case <-tick.C:
			sse.event("ping", map[string]any{"type": "ping"})
		case <-r.Context().Done():
			// 客户端断开：runChat 用的是同一个 ctx，会随之退出；等它归还槽位。
			<-done
			return
		}
	}
	if res.err != nil {
		sse.event("error", map[string]any{
			"type":  "error",
			"error": map[string]any{"type": anthropicErrType(res.err.Status), "message": res.err.Message},
		})
		return
	}

	sse.event("content_block_start", map[string]any{
		"type": "content_block_start", "index": 0,
		"content_block": map[string]any{"type": "text", "text": ""},
	})
	for _, piece := range splitRunes(res.text, 400) {
		sse.event("content_block_delta", map[string]any{
			"type": "content_block_delta", "index": 0,
			"delta": map[string]any{"type": "text_delta", "text": piece},
		})
	}
	sse.event("content_block_stop", map[string]any{"type": "content_block_stop", "index": 0})
	sse.event("message_delta", map[string]any{
		"type":  "message_delta",
		"delta": map[string]any{"stop_reason": "end_turn", "stop_sequence": nil},
		"usage": map[string]any{"output_tokens": estimateTokens(res.text)},
	})
	sse.event("message_stop", map[string]any{"type": "message_stop"})
}

// handleCountTokens 按字数粗估 token 数，不调上游。
func (h *Handler) handleCountTokens(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeAnthropicError(w, http.StatusMethodNotAllowed, "仅支持 POST")
		return
	}
	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeAnthropicError(w, http.StatusBadRequest, "读取请求体失败")
		return
	}
	var areq anthropicRequest
	if err := json.Unmarshal(body, &areq); err != nil {
		writeAnthropicError(w, http.StatusBadRequest, "JSON 解析失败: "+err.Error())
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"input_tokens": estimateAnthropicInput(areq)})
}

// writeAnthropicReply 输出非流式 message。model 回填客户端原样传来的名字。
// 网关在客户端要非流式时直接返回 JSON；流式请求只在本地应答（max_tokens=1）时走这里，
// 这时也按流式事件发，免得客户端解析失败。
func (h *Handler) writeAnthropicReply(w http.ResponseWriter, areq anthropicRequest, text, msgID string, inputTokens int, stop string) {
	if !strings.HasPrefix(msgID, "msg_") {
		msgID = "msg_" + msgID
	}
	usage := map[string]any{"input_tokens": inputTokens, "output_tokens": estimateTokens(text)}
	if areq.Stream {
		sse := newSSE(w)
		sse.event("message_start", map[string]any{
			"type": "message_start",
			"message": map[string]any{
				"id": msgID, "type": "message", "role": "assistant", "model": areq.Model,
				"content": []any{}, "stop_reason": nil, "stop_sequence": nil,
				"usage": map[string]any{"input_tokens": inputTokens, "output_tokens": 0},
			},
		})
		sse.event("content_block_start", map[string]any{"type": "content_block_start", "index": 0, "content_block": map[string]any{"type": "text", "text": ""}})
		sse.event("content_block_delta", map[string]any{"type": "content_block_delta", "index": 0, "delta": map[string]any{"type": "text_delta", "text": text}})
		sse.event("content_block_stop", map[string]any{"type": "content_block_stop", "index": 0})
		sse.event("message_delta", map[string]any{"type": "message_delta", "delta": map[string]any{"stop_reason": stop, "stop_sequence": nil}, "usage": map[string]any{"output_tokens": usage["output_tokens"]}})
		sse.event("message_stop", map[string]any{"type": "message_stop"})
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"id":            msgID,
		"type":          "message",
		"role":          "assistant",
		"model":         areq.Model,
		"content":       []map[string]any{{"type": "text", "text": text}},
		"stop_reason":   stop,
		"stop_sequence": nil,
		"usage":         usage,
	})
}

// anthropicToOpenAIContent 把 Anthropic 的 content 转成 OpenAI 的 content，
// 交给现有的 contentParts 解析。
//
//   - text 块：合并成一段，块之间空一行；Claude Code 自己塞进用户消息的内部记录整块跳过
//     （<system-reminder> 提示、/model 这类本地命令的记录和输出），发给 Sumus 只会浪费积分、干扰回答。
//     如果跳过后什么都不剩，就保留原文，免得 lastUserText 退回去重发更早的消息。
//   - image 块：base64 → image_url data URI；url → image_url。
//   - document 块：base64 → input_file；纯文本来源并进文字。
//   - tool_result 块：取其中的文字，标成「工具结果」并进文字。
//   - tool_use、thinking 等：跳过。
func anthropicToOpenAIContent(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 {
		return raw
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return raw
	}
	var blocks []anthropicBlock
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return raw
	}

	var texts, reminders []string
	var files []map[string]any
	for _, b := range blocks {
		switch b.Type {
		case "text":
			if isClaudeCodeInternal(b.Text) {
				reminders = append(reminders, b.Text)
				continue
			}
			if strings.TrimSpace(b.Text) != "" {
				texts = append(texts, b.Text)
			}
		case "image":
			if b.Source == nil {
				continue
			}
			switch b.Source.Type {
			case "base64":
				files = append(files, map[string]any{"type": "image_url", "image_url": map[string]any{
					"url": "data:" + b.Source.MediaType + ";base64," + b.Source.Data,
				}})
			case "url":
				files = append(files, map[string]any{"type": "image_url", "image_url": map[string]any{"url": b.Source.URL}})
			}
		case "document":
			if b.Source == nil {
				continue
			}
			switch b.Source.Type {
			case "base64":
				name := b.Title
				if name == "" {
					name = "document.pdf"
				}
				files = append(files, map[string]any{"type": "input_file", "input_file": map[string]any{
					"filename": name,
					"data":     "data:" + b.Source.MediaType + ";base64," + b.Source.Data,
				}})
			case "text":
				texts = append(texts, b.Source.Data)
			case "url":
				texts = append(texts, "[文档链接] "+b.Source.URL)
			}
		case "tool_result":
			if t := blockText(b.Content); t != "" {
				texts = append(texts, "[工具结果]\n"+t)
			}
		}
	}
	if len(texts) == 0 && len(files) == 0 {
		texts = reminders
	}

	parts := make([]map[string]any, 0, len(files)+1)
	if len(texts) > 0 {
		parts = append(parts, map[string]any{"type": "text", "text": strings.Join(texts, "\n\n")})
	}
	parts = append(parts, files...)
	out, err := json.Marshal(parts)
	if err != nil {
		return raw
	}
	return out
}

// claudeCodeInternalTags 是 Claude Code 写进用户消息、但不是用户本人说的话的块，按开头标签识别：
// <system-reminder> 是内部提示；其余是 /model 这类本地命令的记录（说明、命令名、命令输出）。
var claudeCodeInternalTags = []string{
	"<system-reminder>",
	"<local-command-caveat>",
	"<command-name>",
	"<command-message>",
	"<local-command-stdout>",
	"<local-command-stderr>",
}

// isClaudeCodeInternal 判断一个 text 块是否整块都是 Claude Code 的内部记录：
// 以上面某个标签开头，并以闭合标签结尾。用户自己打的字即使提到这些标签也不会被误删，
// 因为那种文字一般不会恰好以标签开头又以标签结尾。
func isClaudeCodeInternal(s string) bool {
	t := strings.TrimSpace(s)
	if !strings.HasSuffix(t, ">") {
		return false
	}
	for _, tag := range claudeCodeInternalTags {
		if strings.HasPrefix(t, tag) {
			return true
		}
	}
	return false
}

// blockText 取字符串或块数组里的全部文字（system 字段、tool_result 内容共用）。
func blockText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}
	var blocks []anthropicBlock
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return ""
	}
	var sb strings.Builder
	for _, b := range blocks {
		if b.Type == "text" && b.Text != "" {
			if sb.Len() > 0 {
				sb.WriteString("\n")
			}
			sb.WriteString(b.Text)
		}
	}
	return sb.String()
}

// claudeVersioned 匹配 claude-<系列>-<主>-<次>[-日期]，也接受 claude-<系列>-<主>.<次>。
var claudeVersioned = regexp.MustCompile(`^claude-([a-z]+)-(\d+)[-.](\d+)`)

// mapAnthropicModel 把 Claude Code 常用的模型名映射成 Sumus 的模型 ID：
//
//	claude-sonnet-5-5            → anthropic/claude-sonnet-5.5
//	claude-haiku-5-5-20261001    → anthropic/claude-haiku-5.5
//	claude-fable-5-2[1m]         → anthropic/claude-fable-5.2
//
// 已经带 provider 前缀（含 /）的原样透传，其他认不出的也原样透传。
func mapAnthropicModel(m string) string {
	m = strings.TrimSpace(m)
	if i := strings.Index(m, "["); i >= 0 {
		m = m[:i]
	}
	if m == "" || strings.Contains(m, "/") {
		return m
	}
	if g := claudeVersioned.FindStringSubmatch(m); g != nil {
		return "anthropic/claude-" + g[1] + "-" + g[2] + "." + g[3]
	}
	return m
}

// anthropicConvKey 决定会话粘性的 key。
// 优先 X-Conversation-Id，其次 Claude Code 的会话 ID（请求头或 metadata.user_id），
// 都没有就和 OpenAI 接口一样按首条用户消息派生。
// 末尾拼上模型名：Claude Code 起标题之类的后台请求用的是小模型，
// 分开放，免得混进主对话。
func anthropicConvKey(r *http.Request, areq anthropicRequest, req chatRequest) string {
	key := r.Header.Get("X-Conversation-Id")
	if key == "" {
		key = r.Header.Get("X-Claude-Code-Session-Id")
	}
	if key == "" {
		key = areq.Metadata.UserID
	}
	if key == "" {
		return deriveConvKey(req.Messages)
	}
	return "a-" + key + "|" + req.Model
}

// estimateTokens 粗估 token 数：按 UTF-8 字节数 / 4。
// 英文约 4 字符一个 token，中文约 1.3 字一个 token，两者都大致对得上。
// 上游不返回真实用量，这个数只给客户端显示上下文占用用。
func estimateTokens(s string) int {
	if s == "" {
		return 0
	}
	return (len(s) + 3) / 4
}

func estimateAnthropicInput(areq anthropicRequest) int {
	n := estimateTokens(blockText(areq.System))
	for _, m := range areq.Messages {
		n += estimateTokens(blockText(m.Content))
	}
	return n
}

// splitRunes 按字符数切段，避免切坏多字节字符。
func splitRunes(s string, size int) []string {
	if s == "" {
		return []string{""}
	}
	rs := []rune(s)
	var out []string
	for i := 0; i < len(rs); i += size {
		j := i + size
		if j > len(rs) {
			j = len(rs)
		}
		out = append(out, string(rs[i:j]))
	}
	return out
}

// sseWriter 写 Anthropic 风格的 SSE 事件（event: + data:）。
type sseWriter struct {
	w http.ResponseWriter
	f http.Flusher
}

func newSSE(w http.ResponseWriter) *sseWriter {
	w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)
	f, _ := w.(http.Flusher)
	s := &sseWriter{w: w, f: f}
	if f != nil {
		f.Flush()
	}
	return s
}

func (s *sseWriter) event(name string, payload any) {
	b, _ := json.Marshal(payload)
	fmt.Fprintf(s.w, "event: %s\ndata: %s\n\n", name, b)
	if s.f != nil {
		s.f.Flush()
	}
}

// anthropicErrType 把 HTTP 状态码映射成 Anthropic 的错误类型。
// Claude Code 遇到 overloaded_error / api_error 会自动重试。
func anthropicErrType(status int) string {
	switch status {
	case http.StatusBadRequest, http.StatusMethodNotAllowed:
		return "invalid_request_error"
	case http.StatusUnauthorized:
		return "authentication_error"
	case http.StatusForbidden:
		return "permission_error"
	case http.StatusNotFound:
		return "not_found_error"
	case http.StatusRequestEntityTooLarge:
		return "request_too_large"
	case http.StatusTooManyRequests:
		return "rate_limit_error"
	case http.StatusServiceUnavailable, 529:
		return "overloaded_error"
	default:
		return "api_error"
	}
}

// writeAnthropicError 输出 Anthropic 格式的错误响应。
func writeAnthropicError(w http.ResponseWriter, status int, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"type":  "error",
		"error": map[string]any{"type": anthropicErrType(status), "message": message},
	})
}
