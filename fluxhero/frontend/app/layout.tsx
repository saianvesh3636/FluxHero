import type { Metadata } from 'next';
import './globals.css';
import { ThemeProvider } from '../utils/theme-context';
import ThemeToggle from '../components/ThemeToggle';

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
        <ThemeProvider>
          <div className="theme-toggle-container">
            <ThemeToggle />
          </div>
          {children}
        </ThemeProvider>
      </body>
    </html>
  );
}
