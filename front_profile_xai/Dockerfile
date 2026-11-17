FROM node:20-slim

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

# Para desarrollo solemos usar npm run dev
CMD ["npm", "run", "dev"]