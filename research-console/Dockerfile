FROM node:22-alpine AS dependencies

WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci


FROM node:22-alpine AS builder

WORKDIR /app
ENV NEXT_TELEMETRY_DISABLED=1
ARG NEXT_PUBLIC_API_BASE=http://localhost:8080/api/v1
ENV NEXT_PUBLIC_API_BASE=$NEXT_PUBLIC_API_BASE

COPY --from=dependencies /app/node_modules ./node_modules
COPY app ./app
COPY components ./components
COPY lib ./lib
COPY next.config.js postcss.config.js tailwind.config.js tsconfig.json .eslintrc.json ./
RUN npm run build


FROM node:22-alpine AS runtime

WORKDIR /app
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0

RUN addgroup --system --gid 10001 seatflow \
    && adduser --system --uid 10001 --ingroup seatflow nextjs

COPY package.json package-lock.json next.config.js ./
RUN npm ci --omit=dev --ignore-scripts \
    && npm cache clean --force

COPY --from=builder --chown=nextjs:seatflow /app/.next ./.next

USER nextjs:seatflow

EXPOSE 3000

HEALTHCHECK --interval=15s --timeout=5s --retries=8 --start-period=30s \
  CMD wget --quiet --output-document=- http://127.0.0.1:3000/ >/dev/null

CMD ["npm", "start", "--", "--hostname", "0.0.0.0", "--port", "3000"]
