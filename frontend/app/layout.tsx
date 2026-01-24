import type { Metadata } from 'next';
import './globals.css';
import { WebSocketProvider } from '../contexts/WebSocketContext';
import { TradingModeProvider } from '../contexts/TradingModeContext';
import ErrorBoundary from '../components/ErrorBoundary';
import { AppShell } from '../components/layout';

export const metadata: Metadata = {
  title: 'FluxHero - Adaptive Quant Trading System',
  description: 'Real-time adaptive quantitative trading dashboard',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
      </head>
      <body className="bg-panel-900 text-text-800 antialiased">
        <ErrorBoundary>
          <WebSocketProvider url="/ws/prices">
            <TradingModeProvider>
              <AppShell>
                {children}
              </AppShell>
            </TradingModeProvider>
          </WebSocketProvider>
        </ErrorBoundary>
      </body>
    </html>
  );
}
