import React from 'react';
import { cn } from '../../lib/utils';

export interface TableProps extends React.TableHTMLAttributes<HTMLTableElement> {}

/**
 * Table component with consistent styling
 * Follows design system: panel-700 header, hover states, no borders
 */
export function Table({ className, children, ...props }: TableProps) {
  return (
    <div className="overflow-x-auto">
      <table
        className={cn('w-full border-collapse', className)}
        {...props}
      >
        {children}
      </table>
    </div>
  );
}

export interface TableHeaderProps extends React.HTMLAttributes<HTMLTableSectionElement> {}

export function TableHeader({ className, children, ...props }: TableHeaderProps) {
  return (
    <thead className={cn(className)} {...props}>
      {children}
    </thead>
  );
}

export interface TableBodyProps extends React.HTMLAttributes<HTMLTableSectionElement> {}

export function TableBody({ className, children, ...props }: TableBodyProps) {
  return (
    <tbody className={cn(className)} {...props}>
      {children}
    </tbody>
  );
}

export interface TableRowProps extends React.HTMLAttributes<HTMLTableRowElement> {
  isHoverable?: boolean;
}

export function TableRow({ className, isHoverable = true, children, ...props }: TableRowProps) {
  return (
    <tr
      className={cn(
        isHoverable && 'hover:bg-panel-600',
        className
      )}
      {...props}
    >
      {children}
    </tr>
  );
}

export interface TableHeadProps extends React.ThHTMLAttributes<HTMLTableCellElement> {
  sortable?: boolean;
  sorted?: 'asc' | 'desc' | null;
  onSort?: () => void;
}

export function TableHead({
  className,
  sortable = false,
  sorted = null,
  onSort,
  children,
  ...props
}: TableHeadProps) {
  return (
    <th
      className={cn(
        'bg-panel-700 text-text-400 font-medium text-left px-4 py-3 text-sm',
        sortable && 'cursor-pointer select-none hover:text-text-600',
        className
      )}
      onClick={sortable ? onSort : undefined}
      {...props}
    >
      <div className="flex items-center gap-1">
        {children}
        {sortable && (
          <span className="text-xs">
            {sorted === 'asc' ? '↑' : sorted === 'desc' ? '↓' : '↕'}
          </span>
        )}
      </div>
    </th>
  );
}

export interface TableCellProps extends React.TdHTMLAttributes<HTMLTableCellElement> {
  numeric?: boolean;
}

export function TableCell({ className, numeric = false, children, ...props }: TableCellProps) {
  return (
    <td
      className={cn(
        'px-4 py-3 text-text-700 border-b border-panel-600',
        numeric && 'text-right font-mono tabular-nums',
        className
      )}
      {...props}
    >
      {children}
    </td>
  );
}

export interface TableEmptyProps extends React.HTMLAttributes<HTMLTableRowElement> {
  colSpan: number;
  message?: string;
}

export function TableEmpty({
  colSpan,
  message = 'No data available',
  className,
  ...props
}: TableEmptyProps) {
  return (
    <tr className={className} {...props}>
      <td
        colSpan={colSpan}
        className="px-4 py-12 text-center text-text-400"
      >
        {message}
      </td>
    </tr>
  );
}

export default Table;
