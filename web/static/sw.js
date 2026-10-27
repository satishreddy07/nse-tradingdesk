/**
 * NSE Trading Desk — Progressive Web App Service Worker (v2.0)
 * Real-time financial analytics: Network-first with fresh HTML bypass.
 */

const CACHE_NAME = 'nse-tradingdesk-v2';
const STATIC_ASSETS = [
  '/manifest.json',
  '/static/manifest.json',
  '/static/icon-192.png',
  '/static/icon-512.png',
  '/static/icon.svg',
  'https://cdn.tailwindcss.com',
  'https://unpkg.com/lucide@latest'
];

// Install: Cache critical static shell (NEVER cache dynamic '/' HTML)
self.addEventListener('install', (event) => {
  self.skipWaiting();
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(STATIC_ASSETS).catch((err) => {
        console.warn('PWA Asset precaching warning (non-fatal):', err);
      });
    })
  );
});

// Activate: Evict all previous cache versions immediately
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(
        keys.map((key) => {
          if (key !== CACHE_NAME) {
            console.log('Clearing old PWA cache version:', key);
            return caches.delete(key);
          }
        })
      );
    }).then(() => self.clients.claim())
  );
});

// Fetch: Strictly Network-Only / Network-First for real-time market data & HTML navigations
self.addEventListener('fetch', (event) => {
  const request = event.request;
  const url = new URL(request.url);

  // Dynamic API calls, SSE streams, WebSockets, or mutations: Network only
  if (url.pathname.startsWith('/api/') || url.pathname.startsWith('/health') || request.method !== 'GET') {
    event.respondWith(
      fetch(request).catch(() => {
        return new Response(JSON.stringify({ 
          status: 'OFFLINE', 
          data_unavailable: true, 
          message: 'Network offline. Live market data stream unavailable.' 
        }), {
          status: 503,
          headers: { 'Content-Type': 'application/json' }
        });
      })
    );
    return;
  }

  // HTML page navigation: Always Network First, do NOT store HTML in persistent cache
  if (request.mode === 'navigate' || request.headers.get('accept')?.includes('text/html')) {
    event.respondWith(
      fetch(request).catch(() => {
        return caches.match('/manifest.json') || new Response(
          '<!DOCTYPE html><html><body style="background:#0b0e14;color:#f87171;font-family:sans-serif;padding:2rem;text-align:center;"><h2>Offline</h2><p>Market data system is offline. Please check your internet connection.</p></body></html>',
          { headers: { 'Content-Type': 'text/html' } }
        );
      })
    );
    return;
  }

  // Static assets (images, icons, fonts): Cache-first with background revalidation
  event.respondWith(
    caches.match(request).then((cachedResponse) => {
      if (cachedResponse) {
        fetch(request).then((networkResponse) => {
          if (networkResponse && networkResponse.status === 200) {
            caches.open(CACHE_NAME).then((cache) => cache.put(request, networkResponse));
          }
        }).catch(() => {});
        return cachedResponse;
      }
      return fetch(request).then((response) => {
        if (!response || response.status !== 200 || response.type !== 'basic') {
          return response;
        }
        const clone = response.clone();
        caches.open(CACHE_NAME).then((cache) => cache.put(request, clone));
        return response;
      });
    })
  );
});
