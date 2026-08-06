FROM maven:3.9.11-eclipse-temurin-21 AS builder

WORKDIR /build

COPY pom.xml ./
RUN mvn -B -q -DskipTests dependency:go-offline

COPY src ./src
RUN mvn -B -q -DskipTests package


FROM eclipse-temurin:21-jre AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 seatflow \
    && useradd --system --uid 10001 --gid seatflow --home-dir /app --create-home seatflow

WORKDIR /app
COPY --from=builder --chown=seatflow:seatflow /build/target/*.jar /app/app.jar

USER seatflow:seatflow

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --retries=8 --start-period=45s \
  CMD curl --fail --silent --show-error http://127.0.0.1:8080/actuator/health >/dev/null

ENTRYPOINT ["java", "-jar", "/app/app.jar"]
