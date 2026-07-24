#ifndef ASEVOXEL_WS_SERVER_H
#define ASEVOXEL_WS_SERVER_H

// ============================================================================
// Minimal RFC 6455 WebSocket server for localhost single-client daemon mode
// ============================================================================

#include <cstdint>
#include <cstddef>
#include <vector>

// Cross-platform socket type
#ifdef _WIN32
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #include <winsock2.h>
  #include <ws2tcpip.h>
  typedef SOCKET ws_socket_t;
  static constexpr ws_socket_t WS_INVALID_SOCKET = INVALID_SOCKET;
#else
  typedef int ws_socket_t;
  static constexpr ws_socket_t WS_INVALID_SOCKET = -1;
#endif

struct WsMessage {
    enum Opcode : uint8_t {
        CONTINUATION = 0x0,
        TEXT         = 0x1,
        BINARY       = 0x2,
        CLOSE        = 0x8,
        PING         = 0x9,
        PONG         = 0xA
    };
    Opcode opcode;
    std::vector<uint8_t> data;
};

class WsServer {
public:
    WsServer();
    ~WsServer();

    // Start listening on localhost:port. Returns true on success.
    bool start(uint16_t port);

    // Block until a client connects and completes WebSocket handshake.
    // Returns false if listen socket was closed (server stopping).
    bool acceptClient();

    // Read one WebSocket message. Returns false on disconnect or error.
    bool readMessage(WsMessage& msg);

    // Send a binary frame.
    bool sendBinary(const uint8_t* data, size_t len);

    // Send a text frame.
    bool sendText(const char* data, size_t len);

    // Send a pong frame (echo the ping payload).
    bool sendPong(const uint8_t* data, size_t len);

    // Send a close frame and disconnect the client.
    void closeClient();

    // Stop listening and close all sockets.
    void stop();

    bool hasClient() const { return clientSock_ != WS_INVALID_SOCKET; }
    uint16_t port() const { return port_; }

private:
    ws_socket_t listenSock_;
    ws_socket_t clientSock_;
    uint16_t    port_;

    bool performHandshake();
    bool sendFrame(uint8_t opcode, const uint8_t* data, size_t len);
    bool recvExact(void* buf, size_t len);
    bool sendExact(const void* buf, size_t len);
};

#endif // ASEVOXEL_WS_SERVER_H
