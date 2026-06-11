# Build Stage
FROM node:20-alpine as build-stage

# Set work directory
WORKDIR /app

# Install dependencies
COPY package.json package-lock.json* ./
RUN npm install

# Copy source code and build
COPY . .
RUN npm run build

# Production Stage (Nginx)
FROM nginx:alpine as production-stage

# Copy the built files to Nginx's default public directory
COPY --from=build-stage /app/dist /usr/share/nginx/html

# Copy a custom nginx configuration if needed (optional)
# COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
