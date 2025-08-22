#!/bin/bash
# Generate self-signed SSL certificates for Four-Brain System v2 Secure Tunnel
# Stage 2 Implementation

set -e

echo "🔐 Generating SSL certificates for Four-Brain System v2 Secure Tunnel..."

# Create SSL directory if it doesn't exist
mkdir -p /etc/nginx/ssl

# Generate private key
openssl genrsa -out /etc/nginx/ssl/key.pem 2048

# Generate certificate signing request
openssl req -new -key /etc/nginx/ssl/key.pem -out /etc/nginx/ssl/cert.csr -subj "/C=AU/ST=Victoria/L=Melbourne/O=Four-Brain System/OU=IT Department/CN=localhost"

# Generate self-signed certificate
openssl x509 -req -days 365 -in /etc/nginx/ssl/cert.csr -signkey /etc/nginx/ssl/key.pem -out /etc/nginx/ssl/cert.pem

# Set proper permissions
chmod 600 /etc/nginx/ssl/key.pem
chmod 644 /etc/nginx/ssl/cert.pem

# Clean up CSR
rm /etc/nginx/ssl/cert.csr

echo "✅ SSL certificates generated successfully!"
echo "📁 Certificate: /etc/nginx/ssl/cert.pem"
echo "🔑 Private Key: /etc/nginx/ssl/key.pem"
echo ""
echo "⚠️  Note: These are self-signed certificates for development/testing."
echo "   For production, use Let's Encrypt or proper CA-signed certificates."
