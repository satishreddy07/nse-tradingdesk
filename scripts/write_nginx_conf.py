conf = """server {
    server_name nseuniverse.duckdns.org nseuniverse.duckdns.com;

    client_max_body_size 50M;

    # Drop connections from automated vulnerability scanners instantly with 0 bytes
    location ~* (\\.php|xmlrpc\\.php|wp-admin|wp-content|wp-includes|\\.env|\\.git) {
        return 444;
    }

    # Cache static assets in browser
    location /static/ {
        proxy_pass http://127.0.0.1:8000/static/;
        proxy_set_header Host $host;
        expires 7d;
        add_header Cache-Control "public, max-age=604800, immutable";
    }

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

    listen [::]:443 ssl;
    listen 443 ssl;
    ssl_certificate /etc/letsencrypt/live/nseuniverse.duckdns.org/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/nseuniverse.duckdns.org/privkey.pem;
    include /etc/letsencrypt/options-ssl-nginx.conf;
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem;
}

server {
    listen 80;
    listen [::]:80;
    server_name nseuniverse.duckdns.org nseuniverse.duckdns.com;
    return 301 https://nseuniverse.duckdns.org$request_uri;
}
"""

with open("/etc/nginx/sites-available/nseuniverse.duckdns.org.conf", "w") as f:
    f.write(conf)
print("Config written successfully!")
