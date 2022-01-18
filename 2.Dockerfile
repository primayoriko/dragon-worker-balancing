# call from repo root

FROM ubuntu:20.04 AS build

ENV GOLANG_VERSION 1.13.5
ENV GOPATH /go
ENV PATH $GOPATH/bin:/usr/local/go/bin:$PATH

WORKDIR /go/src/DRAGON

COPY . .

RUN apt update && \
    apt install -y g++ wget make golang-go && \
    make DRAGON

FROM alpine:3.9

COPY --from=build /go/src/DRAGON/bin/DRAGON /usr/bin/DRAGON

CMD ["DRAGON", "-alsologtostderr", "-v=4"]
