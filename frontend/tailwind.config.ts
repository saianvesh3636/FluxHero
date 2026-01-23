import type { Config } from 'tailwindcss';

const config: Config = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // Text colors (light to dark for dark mode)
        text: {
          100: '#716f7a',
          200: '#8F8D98',
          300: '#AEACB7',
          400: '#CCCAD5',
          500: '#eae8f3',
          600: '#EFEEF6',
          700: '#F5F4F9',
          800: '#FAF9FC',
          900: '#ffffff',
        },
        // Panel/Background colors (dark to light)
        panel: {
          100: '#292D3C',
          200: '#272A39',
          300: '#252735',
          400: '#232432',
          500: '#21222F',
          600: '#1E1F2B',
          700: '#1C1C28',
          800: '#1A1924',
          900: '#181621',
        },
        // Profit/Positive colors
        profit: {
          100: '#4ADE80',
          500: '#22C55E',
          900: '#16A34A',
        },
        // Loss/Negative colors
        loss: {
          100: '#F87171',
          500: '#EF4444',
          900: '#DC2626',
        },
        // Primary accent (purple)
        accent: {
          100: '#C04DFE',
          500: '#A549FC',
          900: '#8945FA',
        },
        // Secondary accent (blue)
        blue: {
          100: '#5790FC',
          500: '#3E7AEE',
          900: '#2463E0',
        },
        // Warning (amber/orange)
        warning: {
          100: '#FCD34D',
          500: '#F59E0B',
          900: '#D97706',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'Consolas', 'monospace'],
      },
      fontSize: {
        'xs': ['0.75rem', { lineHeight: '1rem' }],
        'sm': ['0.875rem', { lineHeight: '1.25rem' }],
        'base': ['1rem', { lineHeight: '1.5rem' }],
        'lg': ['1.125rem', { lineHeight: '1.75rem' }],
        'xl': ['1.25rem', { lineHeight: '1.75rem' }],
        '2xl': ['1.5rem', { lineHeight: '2rem' }],
        '3xl': ['1.875rem', { lineHeight: '2.25rem' }],
        '4xl': ['2.25rem', { lineHeight: '2.5rem' }],
      },
      borderRadius: {
        'sm': '4px',
        'DEFAULT': '8px',
        'lg': '12px',
        'xl': '16px',
        '2xl': '22px',
      },
      spacing: {
        '4.5': '1.125rem',
        '5.5': '1.375rem',
        '18': '4.5rem',
        '22': '5.5rem',
      },
      maxWidth: {
        'container': '1280px',
      },
      boxShadow: {
        // No shadows by default - we use color contrast instead
        'none': 'none',
      },
    },
  },
  plugins: [],
};

export default config;
