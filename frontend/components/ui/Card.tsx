import React from 'react';
import { cn } from '../../lib/utils';

export interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'default' | 'elevated' | 'highlighted';
  noPadding?: boolean;
}

/**
 * Card component with consistent styling
 * Follows design system: 22px border radius, 20px padding, no shadows/borders
 */
export function Card({
  className,
  variant = 'default',
  noPadding = false,
  children,
  ...props
}: CardProps) {
  const variantClasses = {
    default: 'bg-panel-600',
    elevated: 'bg-panel-400',
    highlighted: 'bg-panel-100',
  };

  return (
    <div
      className={cn(
        'rounded-2xl',
        variantClasses[variant],
        !noPadding && 'p-5',
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
}

export interface CardHeaderProps extends React.HTMLAttributes<HTMLDivElement> {}

export function CardHeader({ className, children, ...props }: CardHeaderProps) {
  return (
    <div className={cn('mb-4', className)} {...props}>
      {children}
    </div>
  );
}

export interface CardTitleProps extends React.HTMLAttributes<HTMLHeadingElement> {}

export function CardTitle({ className, children, ...props }: CardTitleProps) {
  return (
    <h3 className={cn('text-xl font-semibold text-text-900', className)} {...props}>
      {children}
    </h3>
  );
}

export interface CardDescriptionProps extends React.HTMLAttributes<HTMLParagraphElement> {}

export function CardDescription({ className, children, ...props }: CardDescriptionProps) {
  return (
    <p className={cn('text-sm text-text-400 mt-1', className)} {...props}>
      {children}
    </p>
  );
}

export interface CardContentProps extends React.HTMLAttributes<HTMLDivElement> {}

export function CardContent({ className, children, ...props }: CardContentProps) {
  return (
    <div className={cn(className)} {...props}>
      {children}
    </div>
  );
}

export interface CardFooterProps extends React.HTMLAttributes<HTMLDivElement> {}

export function CardFooter({ className, children, ...props }: CardFooterProps) {
  return (
    <div className={cn('mt-4 pt-4 border-t border-panel-500', className)} {...props}>
      {children}
    </div>
  );
}

export default Card;
