#!/usr/bin/env bash
set -e

echo "=== 1. Configuring Nginx for nseuniverse.duckdns.org ==="
cat << 'EOF' > /etc/nginx/sites-available/nseuniverse.duckdns.org.conf
server {
    listen 80;
    listen [::]:80;
    server_name nseuniverse.duckdns.org nseuniverse.duckdns.com;

    client_max_body_size 50M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400s;
        proxy_send_timeout 86400s;
    }
}
EOF

ln -sf /etc/nginx/sites-available/nseuniverse.duckdns.org.conf /etc/nginx/sites-enabled/nseuniverse.duckdns.org.conf
rm -f /etc/nginx/sites-enabled/default

echo "=== 2. Testing Nginx syntax ==="
nginx -t

echo "=== 3. Reloading Nginx ==="
systemctl reload nginx

echo "=== 4. Requesting Let's Encrypt SSL certificate via Certbot ==="
certbot --nginx -d nseuniverse.duckdns.org --non-interactive --agree-tos --register-unsafely-without-email --redirect || echo "Certbot encountered an issue, will inspect"

echo "=== 5. Stopping and disabling Cloudflare Quick Tunnel ==="
systemctl stop cloudflared || true
systemctl disable cloudflared || true

echo "=== 6. Status of Nginx and NSE Services ==="
systemctl is-active nginx
systemctl is-active nse-tradingdesk
nginx -t

echo "=== 7. Verify local curl via host ==="
curl -s -k -I -H "Host: nseuniverse.duckdns.org" http://127.0.0.1 | head -n 10

echo "=== Setup complete! ==="
