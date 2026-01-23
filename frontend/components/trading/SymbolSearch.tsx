'use client';

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { apiClient, SymbolValidationResponse } from '../../utils/api';

interface SymbolSearchProps {
  value: string;
  onChange: (symbol: string, name?: string) => void;
  placeholder?: string;
  label?: string;
  disabled?: boolean;
  className?: string;
}

/**
 * Symbol search input with autocomplete dropdown.
 * Searches for stocks/ETFs by ticker or company name.
 */
export function SymbolSearch({
  value,
  onChange,
  placeholder = 'Search symbol or company...',
  label = 'Symbol',
  disabled = false,
  className = '',
}: SymbolSearchProps) {
  const [query, setQuery] = useState(value);
  const [results, setResults] = useState<SymbolValidationResponse[]>([]);
  const [isOpen, setIsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(-1);
  const inputRef = useRef<HTMLInputElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const debounceRef = useRef<NodeJS.Timeout | null>(null);

  // Sync external value changes
  useEffect(() => {
    setQuery(value);
  }, [value]);

  // Debounced search function
  const searchSymbols = useCallback(async (searchQuery: string) => {
    if (searchQuery.length < 1) {
      setResults([]);
      setIsOpen(false);
      return;
    }

    setIsLoading(true);
    try {
      const response = await apiClient.searchSymbols(searchQuery, 8);
      setResults(response.results);
      setIsOpen(response.results.length > 0);
      setSelectedIndex(-1);
    } catch (error) {
      console.error('Symbol search error:', error);
      setResults([]);
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Handle input change with debounce
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newQuery = e.target.value.toUpperCase();
    setQuery(newQuery);
    onChange(newQuery); // Update parent immediately for manual entry

    // Debounce API call
    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
    }
    debounceRef.current = setTimeout(() => {
      searchSymbols(newQuery);
    }, 300);
  };

  // Handle result selection
  const handleSelect = (result: SymbolValidationResponse) => {
    setQuery(result.symbol);
    onChange(result.symbol, result.name);
    setIsOpen(false);
    setResults([]);
    inputRef.current?.blur();
  };

  // Handle keyboard navigation
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (!isOpen || results.length === 0) {
      if (e.key === 'Enter') {
        e.preventDefault();
        setIsOpen(false);
      }
      return;
    }

    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        setSelectedIndex((prev) => (prev < results.length - 1 ? prev + 1 : prev));
        break;
      case 'ArrowUp':
        e.preventDefault();
        setSelectedIndex((prev) => (prev > 0 ? prev - 1 : -1));
        break;
      case 'Enter':
        e.preventDefault();
        if (selectedIndex >= 0 && selectedIndex < results.length) {
          handleSelect(results[selectedIndex]);
        } else {
          setIsOpen(false);
        }
        break;
      case 'Escape':
        setIsOpen(false);
        setSelectedIndex(-1);
        break;
    }
  };

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(e.target as Node) &&
        inputRef.current &&
        !inputRef.current.contains(e.target as Node)
      ) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Cleanup debounce on unmount
  useEffect(() => {
    return () => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
    };
  }, []);

  return (
    <div className={`relative ${className}`}>
      {label && (
        <label className="block text-sm font-medium text-text-600 mb-2">
          {label}
        </label>
      )}

      <div className="relative">
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          onFocus={() => query.length >= 1 && results.length > 0 && setIsOpen(true)}
          placeholder={placeholder}
          disabled={disabled}
          className="w-full bg-panel-300 text-text-800 rounded px-4 py-3 border-none focus:outline-none focus:ring-2 focus:ring-accent-500 placeholder-text-300"
          autoComplete="off"
          spellCheck={false}
        />

        {/* Loading indicator */}
        {isLoading && (
          <div className="absolute right-3 top-1/2 -translate-y-1/2">
            <div className="w-4 h-4 border-2 border-accent-500 border-t-transparent rounded-full animate-spin" />
          </div>
        )}
      </div>

      {/* Dropdown results */}
      {isOpen && results.length > 0 && (
        <div
          ref={dropdownRef}
          className="absolute z-50 w-full mt-1 bg-panel-500 border border-panel-300 rounded-lg shadow-lg max-h-64 overflow-y-auto"
        >
          {results.map((result, index) => (
            <button
              key={result.symbol}
              type="button"
              onClick={() => handleSelect(result)}
              onMouseEnter={() => setSelectedIndex(index)}
              className={`w-full px-4 py-3 text-left flex items-center justify-between hover:bg-panel-400 transition-colors ${
                index === selectedIndex ? 'bg-panel-400' : ''
              } ${index === 0 ? 'rounded-t-lg' : ''} ${
                index === results.length - 1 ? 'rounded-b-lg' : ''
              }`}
            >
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="font-semibold text-text-800">{result.symbol}</span>
                  {result.type && (
                    <span className="text-xs px-1.5 py-0.5 bg-panel-300 text-text-400 rounded uppercase">
                      {result.type}
                    </span>
                  )}
                </div>
                <p className="text-sm text-text-400 truncate">{result.name}</p>
              </div>
              {result.exchange && (
                <span className="text-xs text-text-300 ml-2">{result.exchange}</span>
              )}
            </button>
          ))}
        </div>
      )}

      {/* Helper text */}
      <p className="text-xs text-text-300 mt-1">
        Type to search stocks and ETFs
      </p>
    </div>
  );
}

export default SymbolSearch;
