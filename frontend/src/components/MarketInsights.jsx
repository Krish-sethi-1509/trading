import { useCallback, useEffect, useState } from 'react'
import { apiRequest, formatTimestamp } from '../lib/api'

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'events', label: 'Upcoming Events' },
  { id: 'news', label: 'News & Analysis' },
]

function formatPrice(value) {
  return value != null && Number.isFinite(Number(value)) ? '$' + Number(value).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : '—'
}

function formatMove(value) {
  if (value == null || !Number.isFinite(Number(value))) return '—'
  const amount = Number(value)
  return (amount > 0 ? '+' : '') + amount.toFixed(2) + '%'
}

function SourceList({ items, kind }) {
  if (!items?.length) {
    return <div className="insights-empty">No {kind === 'events' ? 'upcoming-event' : 'news'} results were returned. Try again shortly.</div>
  }
  return (
    <div className="insight-source-list">
      {items.map((source, index) => (
        <a className="insight-source-card" href={source.url} target="_blank" rel="noreferrer" key={source.url + '-' + index}>
          <span className="insight-source-meta">{source.domain || 'Source'}{source.published_at ? ' · ' + source.published_at : ''}</span>
          <strong>{source.title}</strong>
          <span>{source.snippet}</span>
          <small>Open source ↗</small>
        </a>
      ))}
      {kind === 'events' && <p className="insight-footnote">Search results may describe event schedules. Confirm release dates and times with the linked source.</p>}
    </div>
  )
}

export default function MarketInsights() {
  const [tab, setTab] = useState('overview')
  const [snapshot, setSnapshot] = useState(null)
  const [snapshotError, setSnapshotError] = useState('')
  const [sources, setSources] = useState({})
  const [sourceErrors, setSourceErrors] = useState({})
  const [loadingSources, setLoadingSources] = useState(false)
  const [brief, setBrief] = useState(null)
  const [briefError, setBriefError] = useState('')
  const [briefLoading, setBriefLoading] = useState(false)

  const loadSnapshot = useCallback(async (signal) => {
    try {
      const result = await apiRequest('/market-insights', { signal })
      setSnapshot(result)
      setSnapshotError('')
    } catch (error) {
      if (error.name !== 'AbortError') setSnapshotError(error.message)
    }
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    loadSnapshot(controller.signal)
    const timer = window.setInterval(() => loadSnapshot(controller.signal), 60_000)
    return () => {
      controller.abort()
      window.clearInterval(timer)
    }
  }, [loadSnapshot])

  useEffect(() => {
    if (tab === 'overview' || sources[tab] || sourceErrors[tab]) return
    let active = true
    setLoadingSources(true)
    apiRequest('/market-news?kind=' + tab)
      .then((result) => { if (active) setSources((current) => ({ ...current, [tab]: result.sources || [] })) })
      .catch((error) => { if (active) setSourceErrors((current) => ({ ...current, [tab]: error.message })) })
      .finally(() => { if (active) setLoadingSources(false) })
    return () => { active = false }
  }, [tab, sources, sourceErrors])

  function refreshSources() {
    setSourceErrors((current) => ({ ...current, [tab]: '' }))
    setSources((current) => { const next = { ...current }; delete next[tab]; return next })
  }

  async function generateBrief() {
    if (!snapshot) return
    setBriefLoading(true)
    setBriefError('')
    try {
      const metrics = [
        'Current stored quote: ' + formatPrice(snapshot.price) + '.',
        'Observed daily change: ' + formatMove(snapshot.daily_change_pct) + '.',
        'Observed day range: low ' + formatPrice(snapshot.day_low) + ', high ' + formatPrice(snapshot.day_high) + '.',
        'Technical horizons: ' + (snapshot.technical_scores || []).map((item) => item.label + ' ' + item.direction + (item.score == null ? '' : ' (' + item.score + '/100)')).join('; ') + '.',
        'Observed 7-day support: ' + formatPrice(snapshot.support) + '; resistance: ' + formatPrice(snapshot.resistance) + '.',
      ].join(' ')
      const result = await apiRequest('/chat', {
        method: 'POST',
        body: JSON.stringify({
          query: 'Write a brief educational AI overview of current gold market context. Explain the supplied quantitative observations neutrally, then summarize recent macro drivers only when supported by retrieved sources. Do not make a trade recommendation. ' + metrics,
          history: [],
        }),
      })
      setBrief(result)
    } catch (error) {
      setBriefError(error.message)
    } finally {
      setBriefLoading(false)
    }
  }

  const dailyMove = Number(snapshot?.daily_change_pct)
  const moveClass = Number.isFinite(dailyMove) && dailyMove > 0 ? 'positive' : dailyMove < 0 ? 'negative' : ''

  return (
    <section className="panel market-insights-panel">
      <div className="panel-heading insights-heading">
        <div>
          <div className="eyebrow">MARKET CONTEXT · XAU/USD</div>
          <h2>Market insights</h2>
        </div>
        <span className="insight-asof">{snapshot?.as_of ? 'Updated ' + formatTimestamp(snapshot.as_of) : 'Waiting for observations'}</span>
      </div>

      <nav className="insight-tabs" aria-label="Market insight views">
        {TABS.map((item) => (
          <button key={item.id} type="button" className={tab === item.id ? 'active' : ''} onClick={() => setTab(item.id)}>{item.label}</button>
        ))}
      </nav>

      {tab === 'overview' && (
        <div className="insight-overview">
          {snapshotError && <div className="inline-error" role="alert">{snapshotError}</div>}
          <article className="ai-overview-card">
            <div className="ai-overview-top"><strong><span>✧</span> Market Overview</strong><span className="brief-badge">{brief?.mode === 'local' ? 'LOCAL RESEARCH · NO LLM' : brief ? 'AI · SOURCE GROUNDED' : 'QUANTITATIVE SNAPSHOT'}</span></div>
            {brief ? (
              <>
                <p>{brief.answer}</p>
                {brief.sources?.length > 0 && <div className="brief-sources">Sources: {brief.sources.map((source, index) => <a key={source.url + '-' + index} href={source.url} target="_blank" rel="noreferrer">[{index + 1}] {source.domain || source.title}</a>)}</div>}
                <small>Generated {formatTimestamp(new Date().toISOString())} · Educational macro context only</small>
              </>
            ) : (
              <>
                <p>{snapshot?.overview || 'Loading quantitative market observations…'}</p>
                <small>Quantitative summary from stored quotes. Generate an overview to add public news headlines; without an LLM, the response is clearly labeled local research.</small>
              </>
            )}
            {briefError && <div className="inline-error" role="alert">{briefError}</div>}
            <button type="button" className="text-action" onClick={generateBrief} disabled={briefLoading || !snapshot}>
              {briefLoading ? 'Searching sources…' : brief ? 'Refresh AI overview' : 'Generate AI overview'} <span>↗</span>
            </button>
          </article>

          <div className="daily-change-section">
            <div className="insight-section-title"><h3>Daily Price Change</h3><span>Since 00:00 UTC</span></div>
            <strong className={'daily-change-value ' + moveClass}>{moveClass === 'positive' ? '↑ ' : moveClass === 'negative' ? '↓ ' : ''}{formatMove(snapshot?.daily_change_pct)}</strong>
            <div className="daily-range-track"><div className={moveClass} style={{ left: Math.max(0, Math.min(100, snapshot?.range_position_pct ?? 50)) + '%' }} /></div>
            <div className="range-labels"><span><strong>{formatPrice(snapshot?.day_low)}</strong> Low</span><span>High <strong>{formatPrice(snapshot?.day_high)}</strong></span></div>
          </div>

          <div className="technical-score-section">
            <div className="insight-section-title"><h3>Technical Score</h3><span>Price-derived · 0–100</span></div>
            <div className="technical-score-list">
              {(snapshot?.technical_scores || []).map((item) => (
                <div className="technical-score-row" key={item.label}>
                  <span>{item.label}</span>
                  <strong className={'tech-' + item.direction.toLowerCase()}>{item.score == null ? 'Insufficient data' : item.direction + ' · ' + item.score}</strong>
                  <div className="technical-score-track"><i className={'tech-fill-' + item.direction.toLowerCase()} style={{ width: (item.score ?? 0) + '%' }} /></div>
                </div>
              ))}
            </div>
          </div>

          <div className="levels-section">
            <div><span>Observed 7-day support</span><strong>{formatPrice(snapshot?.support)}</strong></div>
            <i />
            <div><span>Observed 7-day resistance</span><strong>{formatPrice(snapshot?.resistance)}</strong></div>
          </div>
          <p className="insight-footnote">{snapshot?.data_note || 'Metrics are derived from stored price observations and may differ from broker OHLC candles.'}</p>
        </div>
      )}

      {tab !== 'overview' && (
        <div className="insight-news-panel">
          <div className="insight-section-title">
            <h3>{tab === 'events' ? 'Upcoming macro events' : 'Gold & macro news'}</h3>
            <button type="button" className="text-action refresh-sources" onClick={refreshSources}>Refresh ↻</button>
          </div>
          {loadingSources && <div className="insights-empty">Searching linked sources…</div>}
          {!loadingSources && sourceErrors[tab] && <div className="inline-error" role="alert">{sourceErrors[tab]}</div>}
          {!loadingSources && !sourceErrors[tab] && <SourceList items={sources[tab]} kind={tab} />}
        </div>
      )}
      <p className="insights-disclaimer">Educational decision support only · Not financial advice</p>
    </section>
  )
}
