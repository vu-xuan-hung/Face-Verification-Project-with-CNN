# Stage 1: Build React App
FROM node:18-alpine AS builder

WORKDIR /app

# Copy package.json và cài đặt thư viện
COPY package.json package-lock.json ./
RUN npm install

# Copy source code và build
COPY . .
RUN npm run build

# Stage 2: Serve với Nginx
FROM nginx:alpine

# Copy file build từ stage 1 sang thư mục html của nginx
COPY --from=builder /app/dist /usr/share/nginx/html

# Mở port 80
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
