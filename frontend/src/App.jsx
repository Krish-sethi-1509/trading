import { useEffect, useState } from 'react'
import { apiRequest } from './lib/api'
import LiveChart from './components/LiveChart'
import SessionClocks from './components/SessionClocks'
import PredictionPanel from './components/PredictionPanel'
import AccuracyTracker from './components/AccuracyTracker'
import ChatWidget from './components/ChatWidget'
import MarketInsights from './components/MarketInsights'

function useLivePrice() {
  const [quote, setQuote] = useState(null)
  const [error, setError] = useState('')
  useEffect(() => {
    const controller = new AbortController()
    const load = async () => {
      try {
        const result = await apiRequest('/price/live', { signal: controller.signal })
        setQuote(result)
        setError('')
      } catch (requestError) {
        if (requestError.name !== 'AbortError') setError(requestError.message)
      }
    }
    load()
    // Poll the backend quote provider every 30 seconds without requiring a page reload.
    const timer = window.setInterval(load, 30_000)
    return () => {
      controller.abort()
      window.clearInterval(timer)
    }
  }, [])
  return { quote, error }
}

export default function App() {
  const { quote, error } = useLivePrice()
  const price = quote?.price
  return (
    <main className="app-shell">
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />
      <div className="dashboard-container">
        <header className="topbar">
          <a className="brand" href="#top" aria-label="Aurum Quant home">
            <span className="brand-mark">A</span>
            <span className="brand-name">AURUM<span>QUANT</span></span>
          </a>
          <div className="topbar-right">
            <div className={`connection-state ${quote?.is_stale ? 'stale' : quote ? 'connected' : error ? 'disconnected' : ''}`}><i /> {quote?.is_stale ? 'PRICE STALE' : quote ? 'API CONNECTED' : error ? 'API OFFLINE' : 'CONNECTING'}</div>
            <div className="topbar-divider" />
          <div className="live-price-block">
            <span>XAU/USD · GOLDAPI</span>
            <strong>{Number.isFinite(Number(price)) ? `$${Number(price).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : '—'}</strong>
            <small>{quote?.timestamp ? `Updated ${new Intl.DateTimeFormat(undefined, { timeStyle: 'medium' }).format(new Date(quote.timestamp))}` : 'Waiting for quote'}</small>
          </div>
          </div>
        </header>

        <section className="welcome-row" id="top">
          <div>
            <div className="eyebrow">QUANTITATIVE RESEARCH TERMINAL <span className="welcome-dot">/</span> GOLD</div>
            <h1>Market intelligence<span>.</span></h1>
            <p>Machine learning signals and institutional market context for XAU/USD.</p>
          </div>
          <div className="market-open-chip"><i /> GLOBAL MARKETS <strong>LIVE</strong></div>
        </section>

        {error && <div className="top-error" role="status">Live quote unavailable: {error}</div>}

        <div className="dashboard-grid">
          <div className="main-column">
            <LiveChart />
            <MarketInsights />
            <SessionClocks />
          </div>
          <aside className="side-column">
            <PredictionPanel />
            <AccuracyTracker />
          </aside>
        </div>

        <footer className="footer">
          <span>© {new Date().getFullYear()} Aurum Quant · Dissertation research MVP</span>
          <span>For informational and academic use only</span>
        </footer>
      </div>
      <ChatWidget />
    </main>
  )
}
