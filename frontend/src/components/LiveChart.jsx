import { useEffect, useRef, useState } from 'react'

const TRADINGVIEW_SYMBOL = 'FX_IDC:XAUUSD'

export default function LiveChart() {
  const hostRef = useRef(null)
  const [loadError, setLoadError] = useState('')

  useEffect(() => {
    const host = hostRef.current
    if (!host) return undefined

    setLoadError('')
    host.replaceChildren()

    const widget = document.createElement('div')
    widget.className = 'tradingview-widget-container'
    widget.style.height = '100%'
    widget.style.width = '100%'

    const chart = document.createElement('div')
    chart.className = 'tradingview-widget-container__widget'
    chart.style.height = 'calc(100% - 26px)'
    chart.style.width = '100%'
    widget.append(chart)

    const attribution = document.createElement('div')
    attribution.className = 'tradingview-widget-copyright'
    const link = document.createElement('a')
    link.href = 'https://www.tradingview.com/chart/?symbol=FX_IDC%3AXAUUSD'
    link.target = '_blank'
    link.rel = 'noopener noreferrer'
    link.textContent = 'View XAU/USD on TradingView'
    attribution.append(link)
    widget.append(attribution)

    const script = document.createElement('script')
    script.type = 'text/javascript'
    script.src = 'https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js'
    script.async = true
    script.textContent = JSON.stringify({
      autosize: true,
      symbol: TRADINGVIEW_SYMBOL,
      interval: '15',
      timezone: 'Asia/Kolkata',
      theme: 'dark',
      style: '1',
      locale: 'en',
      withdateranges: true,
      hide_side_toolbar: true,
      hide_top_toolbar: false,
      hide_legend: false,
      allow_symbol_change: false,
      save_image: false,
      calendar: false,
      details: false,
      hotlist: false,
      hide_volume: true,
      backgroundColor: '#0b1019',
      gridColor: 'rgba(148, 163, 184, 0.08)',
      support_host: 'https://www.tradingview.com',
    })
    script.onerror = () => setLoadError('TradingView could not load. Check your internet connection and refresh this page.')

    widget.append(script)
    host.append(widget)

    return () => {
      script.onerror = null
      host.replaceChildren()
    }
  }, [])

  return (
    <section className="panel chart-panel">
      <div className="panel-heading chart-heading">
        <div>
          <div className="eyebrow">LIVE MARKET CHART</div>
          <h2>XAU/USD <span className="muted-heading">· Gold Spot / U.S. Dollar</span></h2>
        </div>
        <div className="chart-meta">
          <span className="live-indicator"><i /> TRADINGVIEW FEED</span>
        </div>
      </div>
      <div className="chart-wrap tradingview-chart-wrap">
        <div ref={hostRef} className="tradingview-chart-host" aria-label="Live TradingView XAU/USD chart with historical ranges" />
        {loadError && <div className="chart-widget-error" role="status">{loadError}</div>}
      </div>
      <div className="chart-footer">
        <span>TradingView chart · Choose intervals and date ranges in the chart toolbar</span>
        <span>Forex data may vary by provider</span>
      </div>
    </section>
  )
}
