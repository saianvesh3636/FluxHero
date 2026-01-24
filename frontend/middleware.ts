/**
 * Next.js Middleware for route redirects
 *
 * Redirects:
 * - /live -> /trades?mode=live
 * - /live/analysis -> /trades/analysis?mode=live
 * - /history -> /trades
 *
 * This ensures backwards compatibility after consolidating
 * the live and history pages into the unified /trades page.
 */

import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

export function middleware(request: NextRequest) {
  const url = request.nextUrl;

  // Redirect /live to /trades?mode=live
  if (url.pathname === '/live') {
    const newUrl = new URL('/trades', request.url);
    newUrl.searchParams.set('mode', 'live');
    return NextResponse.redirect(newUrl);
  }

  // Redirect /live/analysis to /trades/analysis?mode=live
  if (url.pathname === '/live/analysis') {
    const newUrl = new URL('/trades/analysis', request.url);
    newUrl.searchParams.set('mode', 'live');
    return NextResponse.redirect(newUrl);
  }

  // Redirect /history to /trades (maintains current mode)
  if (url.pathname === '/history') {
    const newUrl = new URL('/trades', request.url);
    // Preserve existing mode if any
    const existingMode = url.searchParams.get('mode');
    if (existingMode) {
      newUrl.searchParams.set('mode', existingMode);
    }
    return NextResponse.redirect(newUrl);
  }

  return NextResponse.next();
}

export const config = {
  matcher: ['/live', '/live/analysis', '/history'],
};
