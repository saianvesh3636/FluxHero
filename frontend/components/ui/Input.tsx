import React from 'react';
import { cn } from '../../lib/utils';

export interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
  helperText?: string;
}

/**
 * Input component with consistent styling
 * Follows design system: panel-300 background, no border, focus ring
 */
export const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, label, error, helperText, id, ...props }, ref) => {
    const inputId = id || label?.toLowerCase().replace(/\s+/g, '-');

    return (
      <div className="w-full">
        {label && (
          <label
            htmlFor={inputId}
            className="block text-sm font-medium text-text-600 mb-2"
          >
            {label}
          </label>
        )}
        <input
          id={inputId}
          ref={ref}
          className={cn(
            'w-full bg-panel-300 text-text-800 placeholder:text-text-300',
            'rounded px-4 py-3 border-none',
            'focus:outline-none focus:ring-2 focus:ring-accent-500',
            'disabled:opacity-50 disabled:cursor-not-allowed',
            error && 'ring-2 ring-loss-500',
            className
          )}
          {...props}
        />
        {error && (
          <p className="mt-1 text-sm text-loss-500">{error}</p>
        )}
        {helperText && !error && (
          <p className="mt-1 text-sm text-text-300">{helperText}</p>
        )}
      </div>
    );
  }
);

Input.displayName = 'Input';

export default Input;
