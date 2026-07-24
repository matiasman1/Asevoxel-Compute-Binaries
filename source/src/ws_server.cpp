// ============================================================================
// Minimal RFC 6455 WebSocket server — single-client, localhost, no TLS
// ============================================================================

#include "ws_server.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <algorithm>

// ============================================================================
// Platform socket helpers
// ============================================================================

#ifdef _WIN32
  #ifdef _MSC_VER
    #pragma comment(lib, "ws2_32")
  #endif
  static void ws_socket_init() {
      static bool done = false;
      if (!done) { WSADATA d; WSAStartup(MAKEWORD(2,2), &d); done = true; }
  }
  static void ws_close_socket(ws_socket_t s) { closesocket(s); }
  static int ws_recv(ws_socket_t s, void* buf, int len) { return recv(s, (char*)buf, len, 0); }
  static int ws_send(ws_socket_t s, const void* buf, int len) { return send(s, (const char*)buf, len, 0); }
#else
  #include <sys/socket.h>
  #include <netinet/in.h>
  #include <netinet/tcp.h>
  #include <arpa/inet.h>
  #include <unistd.h>
  #include <signal.h>
  static void ws_socket_init() {
      static bool done = false;
      if (!done) { signal(SIGPIPE, SIG_IGN); done = true; }
  }
  static void ws_close_socket(ws_socket_t s) { close(s); }
  static int ws_recv(ws_socket_t s, void* buf, int len) { return recv(s, buf, (size_t)len, 0); }
  static int ws_send(ws_socket_t s, const void* buf, int len) { return send(s, buf, (size_t)len, MSG_NOSIGNAL); }
#endif

// ============================================================================
// Minimal SHA-1 (for Sec-WebSocket-Accept)
// ============================================================================

static uint32_t rotl32(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }

static void sha1(const uint8_t* data, size_t len, uint8_t out[20]) {
    uint32_t h0 = 0x67452301, h1 = 0xEFCDAB89, h2 = 0x98BADCFE,
             h3 = 0x10325476, h4 = 0xC3D2E1F0;

    size_t paddedLen = ((len + 8) / 64 + 1) * 64;
    std::vector<uint8_t> buf(paddedLen, 0);
    memcpy(buf.data(), data, len);
    buf[len] = 0x80;

    uint64_t bitLen = (uint64_t)len * 8;
    for (int i = 0; i < 8; i++)
        buf[paddedLen - 1 - i] = (uint8_t)(bitLen >> (i * 8));

    for (size_t off = 0; off < paddedLen; off += 64) {
        uint32_t w[80];
        for (int i = 0; i < 16; i++)
            w[i] = ((uint32_t)buf[off+i*4] << 24) | ((uint32_t)buf[off+i*4+1] << 16) |
                   ((uint32_t)buf[off+i*4+2] << 8) | (uint32_t)buf[off+i*4+3];
        for (int i = 16; i < 80; i++)
            w[i] = rotl32(w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16], 1);

        uint32_t a = h0, b = h1, c = h2, d = h3, e = h4;
        for (int i = 0; i < 80; i++) {
            uint32_t f, k;
            if      (i < 20) { f = (b & c) | (~b & d);            k = 0x5A827999; }
            else if (i < 40) { f = b ^ c ^ d;                      k = 0x6ED9EBA1; }
            else if (i < 60) { f = (b & c) | (b & d) | (c & d);   k = 0x8F1BBCDC; }
            else              { f = b ^ c ^ d;                      k = 0xCA62C1D6; }
            uint32_t temp = rotl32(a, 5) + f + e + k + w[i];
            e = d; d = c; c = rotl32(b, 30); b = a; a = temp;
        }
        h0 += a; h1 += b; h2 += c; h3 += d; h4 += e;
    }

    for (int i = 0; i < 4; i++) {
        out[i]    = (h0 >> (24 - i*8)) & 0xFF;
        out[4+i]  = (h1 >> (24 - i*8)) & 0xFF;
        out[8+i]  = (h2 >> (24 - i*8)) & 0xFF;
        out[12+i] = (h3 >> (24 - i*8)) & 0xFF;
        out[16+i] = (h4 >> (24 - i*8)) & 0xFF;
    }
}

static std::string base64_encode(const uint8_t* data, size_t len) {
    static const char t[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((len + 2) / 3 * 4);
    for (size_t i = 0; i < len; i += 3) {
        uint32_t n = (uint32_t)data[i] << 16;
        if (i+1 < len) n |= (uint32_t)data[i+1] << 8;
        if (i+2 < len) n |= (uint32_t)data[i+2];
        out += t[(n >> 18) & 0x3F];
        out += t[(n >> 12) & 0x3F];
        out += (i+1 < len) ? t[(n >> 6) & 0x3F] : '=';
        out += (i+2 < len) ? t[n & 0x3F] : '=';
    }
    return out;
}

// ============================================================================
// WebSocket server implementation
// ============================================================================

WsServer::WsServer()
    : listenSock_(WS_INVALID_SOCKET)
    , clientSock_(WS_INVALID_SOCKET)
    , port_(0)
{
    ws_socket_init();
}

WsServer::~WsServer() {
    stop();
}

bool WsServer::recvExact(void* buf, size_t len) {
    auto* p = static_cast<uint8_t*>(buf);
    size_t total = 0;
    while (total < len) {
        int n = ws_recv(clientSock_, p + total, (int)(len - total));
        if (n <= 0) return false;
        total += (size_t)n;
    }
    return true;
}

bool WsServer::sendExact(const void* buf, size_t len) {
    auto* p = static_cast<const uint8_t*>(buf);
    size_t total = 0;
    while (total < len) {
        int n = ws_send(clientSock_, p + total, (int)(len - total));
        if (n <= 0) return false;
        total += (size_t)n;
    }
    return true;
}

bool WsServer::start(uint16_t port) {
    listenSock_ = socket(AF_INET, SOCK_STREAM, 0);
    if (listenSock_ == WS_INVALID_SOCKET) return false;

    // Allow port reuse for quick restart
    int opt = 1;
    setsockopt(listenSock_, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK); // localhost only
    addr.sin_port = htons(port);

    if (bind(listenSock_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        ws_close_socket(listenSock_);
        listenSock_ = WS_INVALID_SOCKET;
        return false;
    }

    if (listen(listenSock_, 1) < 0) {
        ws_close_socket(listenSock_);
        listenSock_ = WS_INVALID_SOCKET;
        return false;
    }

    // Retrieve actual bound port (useful if port was 0)
    struct sockaddr_in bound{};
    socklen_t blen = sizeof(bound);
    if (getsockname(listenSock_, (struct sockaddr*)&bound, &blen) == 0) {
        port_ = ntohs(bound.sin_port);
    } else {
        port_ = port;
    }

    return true;
}

bool WsServer::acceptClient() {
    if (listenSock_ == WS_INVALID_SOCKET) return false;

    clientSock_ = accept(listenSock_, nullptr, nullptr);
    if (clientSock_ == WS_INVALID_SOCKET) return false;

    // Disable Nagle for lower latency
    int opt = 1;
    setsockopt(clientSock_, IPPROTO_TCP, TCP_NODELAY, (const char*)&opt, sizeof(opt));

    if (!performHandshake()) {
        closeClient();
        return false;
    }
    return true;
}

// ============================================================================
// HTTP upgrade handshake
// ============================================================================

static std::string extract_header(const std::string& req, const std::string& name) {
    std::string lower_name = name;
    std::string lower_req = req;
    std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);
    std::transform(lower_req.begin(), lower_req.end(), lower_req.begin(), ::tolower);

    size_t pos = lower_req.find(lower_name + ":");
    if (pos == std::string::npos) return "";

    // Use original request (not lowered) for the value
    size_t valStart = pos + name.size() + 1;
    while (valStart < req.size() && (req[valStart] == ' ' || req[valStart] == '\t'))
        valStart++;
    size_t valEnd = req.find("\r\n", valStart);
    if (valEnd == std::string::npos) valEnd = req.size();
    return req.substr(valStart, valEnd - valStart);
}

bool WsServer::performHandshake() {
    // Read HTTP request until \r\n\r\n
    std::string request;
    request.reserve(4096);
    char buf;
    int attempts = 0;
    while (attempts++ < 8192) {
        if (!recvExact(&buf, 1)) return false;
        request += buf;
        if (request.size() >= 4 &&
            request.substr(request.size()-4) == "\r\n\r\n") break;
    }
    if (request.find("\r\n\r\n") == std::string::npos) return false;

    // Extract Sec-WebSocket-Key
    std::string key = extract_header(request, "Sec-WebSocket-Key");
    if (key.empty()) return false;

    // Compute Sec-WebSocket-Accept: SHA1(key + GUID), base64 encoded
    static const char* GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    std::string magic = key + GUID;
    uint8_t hash[20];
    sha1((const uint8_t*)magic.c_str(), magic.size(), hash);
    std::string acceptVal = base64_encode(hash, 20);

    // Send 101 Switching Protocols
    std::string response =
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        "Sec-WebSocket-Accept: " + acceptVal + "\r\n"
        "\r\n";
    return sendExact(response.data(), response.size());
}

// ============================================================================
// WebSocket frame read/write (RFC 6455)
// ============================================================================

bool WsServer::readMessage(WsMessage& msg) {
    msg.data.clear();

    uint8_t header[2];
    if (!recvExact(header, 2)) return false;

    bool fin  = (header[0] >> 7) & 1;
    msg.opcode = static_cast<WsMessage::Opcode>(header[0] & 0x0F);
    bool mask = (header[1] >> 7) & 1;
    uint64_t payloadLen = header[1] & 0x7F;

    if (payloadLen == 126) {
        uint8_t ext[2];
        if (!recvExact(ext, 2)) return false;
        payloadLen = ((uint64_t)ext[0] << 8) | ext[1];
    } else if (payloadLen == 127) {
        uint8_t ext[8];
        if (!recvExact(ext, 8)) return false;
        payloadLen = 0;
        for (int i = 0; i < 8; i++)
            payloadLen = (payloadLen << 8) | ext[i];
    }

    uint8_t maskKey[4] = {};
    if (mask) {
        if (!recvExact(maskKey, 4)) return false;
    }

    // Read payload
    msg.data.resize(payloadLen);
    if (payloadLen > 0) {
        if (!recvExact(msg.data.data(), payloadLen)) return false;
        // Unmask
        if (mask) {
            for (size_t i = 0; i < payloadLen; i++)
                msg.data[i] ^= maskKey[i & 3];
        }
    }

    // Handle continuation frames (accumulate until FIN)
    if (!fin && msg.opcode != WsMessage::CLOSE) {
        WsMessage::Opcode origOp = msg.opcode;
        while (!fin) {
            uint8_t hdr2[2];
            if (!recvExact(hdr2, 2)) return false;
            fin = (hdr2[0] >> 7) & 1;
            bool mask2 = (hdr2[1] >> 7) & 1;
            uint64_t len2 = hdr2[1] & 0x7F;
            if (len2 == 126) {
                uint8_t e2[2]; if (!recvExact(e2, 2)) return false;
                len2 = ((uint64_t)e2[0] << 8) | e2[1];
            } else if (len2 == 127) {
                uint8_t e8[8]; if (!recvExact(e8, 8)) return false;
                len2 = 0; for (int i = 0; i < 8; i++) len2 = (len2 << 8) | e8[i];
            }
            uint8_t mk2[4] = {};
            if (mask2) { if (!recvExact(mk2, 4)) return false; }
            size_t prevSize = msg.data.size();
            msg.data.resize(prevSize + len2);
            if (len2 > 0) {
                if (!recvExact(msg.data.data() + prevSize, len2)) return false;
                if (mask2) {
                    for (size_t i = 0; i < len2; i++)
                        msg.data[prevSize + i] ^= mk2[i & 3];
                }
            }
        }
        msg.opcode = origOp;
    }

    return true;
}

bool WsServer::sendFrame(uint8_t opcode, const uint8_t* data, size_t len) {
    // Server→client: no mask
    uint8_t header[10];
    size_t headerLen = 2;
    header[0] = 0x80 | (opcode & 0x0F); // FIN + opcode

    if (len < 126) {
        header[1] = (uint8_t)len;
    } else if (len < 65536) {
        header[1] = 126;
        header[2] = (uint8_t)(len >> 8);
        header[3] = (uint8_t)(len & 0xFF);
        headerLen = 4;
    } else {
        header[1] = 127;
        for (int i = 0; i < 8; i++)
            header[2 + i] = (uint8_t)(len >> ((7 - i) * 8));
        headerLen = 10;
    }

    if (!sendExact(header, headerLen)) return false;
    if (len > 0 && !sendExact(data, len)) return false;
    return true;
}

bool WsServer::sendBinary(const uint8_t* data, size_t len) {
    return sendFrame(0x02, data, len);
}

bool WsServer::sendText(const char* data, size_t len) {
    return sendFrame(0x01, (const uint8_t*)data, len);
}

bool WsServer::sendPong(const uint8_t* data, size_t len) {
    return sendFrame(0x0A, data, len);
}

void WsServer::closeClient() {
    if (clientSock_ != WS_INVALID_SOCKET) {
        // Send close frame (best-effort)
        uint8_t closePayload[2] = { 0x03, 0xE8 }; // 1000 = normal closure
        sendFrame(0x08, closePayload, 2);
        ws_close_socket(clientSock_);
        clientSock_ = WS_INVALID_SOCKET;
    }
}

void WsServer::stop() {
    closeClient();
    if (listenSock_ != WS_INVALID_SOCKET) {
        ws_close_socket(listenSock_);
        listenSock_ = WS_INVALID_SOCKET;
    }
}
