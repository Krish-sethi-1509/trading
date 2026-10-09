import { useEffect, useRef, useState } from 'react'
import { apiRequest, formatTimestamp } from '../lib/api'

const STARTER_PROMPTS = [
  'What recent macro news is affecting gold?',
  'How could real yields influence XAU/USD?',
]

export default function ChatWidget() {
  const [open, setOpen] = useState(false)
  const [messages, setMessages] = useState([])
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const scrollRef = useRef(null)
  const inputRef = useRef(null)

  useEffect(() => {
    if (open) inputRef.current?.focus()
  }, [open])

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages, loading, open])

  async function submitQuestion(text = query) {
    const clean = text.trim()
    if (!clean || loading) return
    const preceding = messages
      .filter((message) => message.role === 'user' || message.role === 'assistant')
      .slice(-8)
      .map(({ role, content }) => ({ role, content }))
    const userMessage = { id: crypto.randomUUID(), role: 'user', content: clean, createdAt: new Date().toISOString() }
    setMessages((current) => [...current, userMessage])
    setQuery('')
    setError('')
    setLoading(true)
    try {
      const result = await apiRequest('/chat', {
        method: 'POST',
        body: JSON.stringify({ query: clean, history: preceding }),
      })
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: result.answer,
          sources: result.sources || [],
          mode: result.mode || 'provider',
          createdAt: new Date().toISOString(),
        },
      ])
    } catch (requestError) {
      setError(requestError.message)
    } finally {
      setLoading(false)
    }
  }

  function handleSubmit(event) {
    event.preventDefault()
    submitQuestion()
  }

  return (
    <div className="chat-widget">
      {open && (
        <section className="chat-window" aria-label="Aurum Macro Assistant">
          <header className="chat-header">
            <div className="chat-avatar">A</div>
            <div className="chat-heading-copy">
              <strong>Aurum Macro Assistant</strong>
              <span><i /> Macro context · source grounded</span>
            </div>
            <button className="chat-close" type="button" aria-label="Close chat" onClick={() => setOpen(false)}>×</button>
          </header>

          <div className="chat-messages" ref={scrollRef} aria-live="polite">
            {!messages.length && (
              <div className="chat-welcome">
                <div className="chat-welcome-mark">✳</div>
                <h3>Macro, in context.</h3>
                <p>Ask about gold, inflation, real yields, or central-bank news. I’ll ground current-event answers in retrieved sources.</p>
                <div className="starter-prompts">
                  {STARTER_PROMPTS.map((prompt) => (
                    <button type="button" key={prompt} onClick={() => submitQuestion(prompt)} disabled={loading}>{prompt}<span>↗</span></button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((message) => (
              <article className={`chat-message ${message.role}`} key={message.id}>
                {message.role === 'assistant' && <div className="message-avatar">A</div>}
                <div className="message-body">
                  {message.role === 'assistant' && <span className="assistant-mode-badge">{message.mode === 'local' ? 'LOCAL RESEARCH · NO LLM' : 'AI · SOURCE GROUNDED'}</span>}
                  <div className="message-bubble">{message.content}</div>
                  {message.sources?.length > 0 && (
                    <div className="source-reference-list">
                      <span className="sources-heading">SOURCES</span>
                      {message.sources.map((source, index) => (
                        <a className="source-reference" href={source.url} target="_blank" rel="noreferrer" key={`${source.url}-${index}`} title={`${source.title}${source.published_at ? ` · ${source.published_at}` : ''}`}>
                          <span>[{index + 1}]</span> {source.domain || source.title}
                        </a>
                      ))}
                    </div>
                  )}
                  <time className="message-time">{formatTimestamp(message.createdAt, { dateStyle: 'none' })}</time>
                </div>
              </article>
            ))}

            {loading && (
              <article className="chat-message assistant">
                <div className="message-avatar">A</div>
                <div className="message-body"><div className="message-bubble typing-bubble"><i /><i /><i /><span>Searching recent macro sources</span></div></div>
              </article>
            )}
          </div>

          {error && <div className="chat-error" role="alert">{error}</div>}
          <form className="chat-composer" onSubmit={handleSubmit}>
            <textarea
              ref={inputRef}
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault()
                  handleSubmit(event)
                }
              }}
              placeholder="Ask about gold or macro…"
              rows={1}
              maxLength={1200}
              aria-label="Ask the macro assistant"
            />
            <button type="submit" aria-label="Send question" disabled={loading || !query.trim()}>↑</button>
          </form>
          <div className="chat-safety-note">Educational macro context only · Not financial advice</div>
        </section>
      )}
      <button className={`chat-launcher ${open ? 'launcher-open' : ''}`} type="button" onClick={() => setOpen((value) => !value)} aria-label={open ? 'Close assistant' : 'Open macro assistant'}>
        {open ? <span>×</span> : <><span className="launcher-spark">✳</span><span className="launcher-label">ASK MACRO AI</span></>}
      </button>
    </div>
  )
}
