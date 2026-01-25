/**
 * Analytics Page Redirect
 *
 * This page redirects to the new Market Research page at /analysis/research
 * Keeping for backwards compatibility with any existing links/bookmarks
 */

'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';

export default function AnalyticsPageRedirect() {
  const router = useRouter();

  useEffect(() => {
    router.replace('/analysis/research');
  }, [router]);

  return (
    <div className="min-h-screen flex items-center justify-center">
      <p className="text-text-400">Redirecting to Market Research...</p>
    </div>
  );
}
