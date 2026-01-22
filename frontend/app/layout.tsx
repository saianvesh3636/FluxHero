import type { Metadata } from 'next';
import './globals.css';
import { ThemeProvider } from '../utils/theme-context';
import { WebSocketProvider } from '../contexts/WebSocketContext';
import ThemeToggle from '../components/ThemeToggle';
import ErrorBoundary from '../components/ErrorBoundary';

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
    <html lang="en">
      <body>
        <ErrorBoundary>
          <ThemeProvider>
            <WebSocketProvider url="/ws/prices">
              <div className="theme-toggle-container">
                <ThemeToggle />
              </div>
              {children}
            </WebSocketProvider>
          </ThemeProvider>
        </ErrorBoundary>
      </body>
    </html>
  );
}
