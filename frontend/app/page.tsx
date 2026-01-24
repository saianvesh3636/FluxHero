import { redirect } from 'next/navigation';

/**
 * Homepage redirects to /trades
 * The trades page is the main entry point for the application
 */
export default function Home() {
  redirect('/trades');
}
