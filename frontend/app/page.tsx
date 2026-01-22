export default function Home() {
  return (
    <main style={{ padding: '2rem', maxWidth: '1200px', margin: '0 auto' }}>
      <h1 style={{ fontSize: '2.5rem', marginBottom: '1rem' }}>
        FluxHero Trading System
      </h1>
      <p style={{ fontSize: '1.2rem', marginBottom: '2rem', color: '#666' }}>
        Adaptive retail quantitative trading platform
      </p>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
          gap: '1.5rem',
          marginTop: '2rem',
        }}
      >
        <div
          style={{
            padding: '1.5rem',
            background: 'white',
            borderRadius: '8px',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
          }}
        >
          <h2 style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>
            Live Trading
          </h2>
          <p style={{ color: '#666' }}>
            Monitor open positions and real-time P&L
          </p>
        </div>

        <div
          style={{
            padding: '1.5rem',
            background: 'white',
            borderRadius: '8px',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
          }}
        >
          <h2 style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>
            Analytics
          </h2>
          <p style={{ color: '#666' }}>
            Charts, indicators, and performance metrics
          </p>
        </div>

        <div
          style={{
            padding: '1.5rem',
            background: 'white',
            borderRadius: '8px',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
          }}
        >
          <h2 style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>
            Trade History
          </h2>
          <p style={{ color: '#666' }}>View past trades and export data</p>
        </div>

        <div
          style={{
            padding: '1.5rem',
            background: 'white',
            borderRadius: '8px',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
          }}
        >
          <h2 style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>
            Backtesting
          </h2>
          <p style={{ color: '#666' }}>Test strategies on historical data</p>
        </div>
      </div>

      <div
        style={{
          marginTop: '3rem',
          padding: '1.5rem',
          background: 'white',
          borderRadius: '8px',
          boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
        }}
      >
        <h3 style={{ fontSize: '1.3rem', marginBottom: '1rem' }}>
          System Status
        </h3>
        <p style={{ color: '#666' }}>
          Backend API integration configured. Ready for implementation.
        </p>
      </div>
    </main>
  );
}
