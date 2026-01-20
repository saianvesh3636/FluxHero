import type { Metadata } from 'next';
import './globals.css';

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
      <body>{children}</body>
    </html>
  );
}
