#ifndef _IN_CSP_CORE_QUEUEBLOCKINGWAIT_H
#define _IN_CSP_CORE_QUEUEBLOCKINGWAIT_H

// Windows: Include winsock2.h for FdWaiter socket pair implementation
// WIN32_LEAN_AND_MEAN is defined project-wide to prevent winsock.h/winsock2.h conflicts
#ifdef _WIN32
#include <winsock2.h>
#pragma comment(lib, "ws2_32.lib")
#endif

#include <mutex>
#include <condition_variable>
#include <csp/core/Time.h>
#include <csp/core/System.h>

#ifdef __linux__
#include <sys/eventfd.h>
#include <unistd.h>
#elif defined(__APPLE__)
#include <unistd.h>
#include <fcntl.h>
#endif

namespace csp
{

class TimeDelta;

class QueueWaiter
{
public:
    QueueWaiter() : m_eventsPending( false )
    {}

    void notify()
    {
        std::lock_guard<std::mutex> guard( m_lock );
        if( !m_eventsPending )
            m_condition.notify_one();
        m_eventsPending = true;

    }

    bool wait( TimeDelta maxWaitTime )
    {
        std::unique_lock<std::mutex> lock( m_lock );
        if( !m_eventsPending && maxWaitTime.asNanoseconds() > 0 )
            m_condition.wait_for( lock, std::chrono::nanoseconds( maxWaitTime.asNanoseconds() ), [this]() { return m_eventsPending; } );

        bool rv = m_eventsPending;
        m_eventsPending = false;
        return rv;
    }

private:
    std::mutex              m_lock;
    std::condition_variable m_condition;
    bool                    m_eventsPending;
};

// FdWaiter provides file descriptor based signaling for integration with
// external event loops like asyncio. The read fd can be registered with
// select/poll/epoll and will become readable when notify() is called.
class FdWaiter
{
public:
    FdWaiter()
    {
#ifdef __linux__
        // Linux: use eventfd (single fd, most efficient)
        m_eventfd = eventfd( 0, EFD_NONBLOCK | EFD_CLOEXEC );
        m_readFd = m_eventfd;
        m_writeFd = m_eventfd;
#elif defined(__APPLE__)
        // macOS: use pipe
        int fds[2];
        if( pipe( fds ) == 0 )
        {
            m_readFd = fds[0];
            m_writeFd = fds[1];
            // Set non-blocking
            fcntl( m_readFd, F_SETFL, O_NONBLOCK );
            fcntl( m_writeFd, F_SETFL, O_NONBLOCK );
        }
        else
        {
            m_readFd = -1;
            m_writeFd = -1;
        }
#elif defined(_WIN32)
        // Windows: use socket pair (localhost loopback)
        m_readFd = INVALID_SOCKET;
        m_writeFd = INVALID_SOCKET;
        createSocketPair();
#endif
    }

    ~FdWaiter()
    {
#ifdef __linux__
        if( m_eventfd >= 0 )
            close( m_eventfd );
#elif defined(__APPLE__)
        if( m_readFd >= 0 )
            close( m_readFd );
        if( m_writeFd >= 0 )
            close( m_writeFd );
#elif defined(_WIN32)
        if( m_readFd != INVALID_SOCKET )
            closesocket( m_readFd );
        if( m_writeFd != INVALID_SOCKET )
            closesocket( m_writeFd );
#endif
    }

    // Get the file descriptor for select/poll registration
    // Returns -1 (or INVALID_SOCKET on Windows) if not available
#ifdef _WIN32
    SOCKET readFd() const { return m_readFd; }
#else
    int readFd() const { return m_readFd; }
#endif

    // Signal the fd (makes it readable)
    void notify()
    {
        std::lock_guard<std::mutex> guard( m_lock );
        if( m_notified )
            return;  // Already notified, avoid filling buffer

        m_notified = true;

#ifdef __linux__
        uint64_t val = 1;
        [[maybe_unused]] auto rv = write( m_eventfd, &val, sizeof( val ) );
#elif defined(__APPLE__)
        char c = 1;
        [[maybe_unused]] auto rv = write( m_writeFd, &c, 1 );
#elif defined(_WIN32)
        char c = 1;
        send( m_writeFd, &c, 1, 0 );
#endif
    }

    // Clear the notification (call after processing)
    void clear()
    {
        std::lock_guard<std::mutex> guard( m_lock );
        m_notified = false;

#ifdef __linux__
        uint64_t val;
        [[maybe_unused]] auto rv = read( m_eventfd, &val, sizeof( val ) );
#elif defined(__APPLE__)
        char buf[64];
        while( read( m_readFd, buf, sizeof( buf ) ) > 0 ) {}
#elif defined(_WIN32)
        char buf[64];
        while( recv( m_readFd, buf, sizeof( buf ), 0 ) > 0 ) {}
#endif
    }

    bool isValid() const
    {
#ifdef _WIN32
        return m_readFd != INVALID_SOCKET;
#else
        return m_readFd >= 0;
#endif
    }

private:
#ifdef _WIN32
    void createSocketPair()
    {
        // Create a listening socket on localhost
        SOCKET listener = socket( AF_INET, SOCK_STREAM, IPPROTO_TCP );
        if( listener == INVALID_SOCKET )
            return;

        struct sockaddr_in addr;
        memset( &addr, 0, sizeof( addr ) );
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl( INADDR_LOOPBACK );
        addr.sin_port = 0;  // Let OS pick a port

        if( bind( listener, (struct sockaddr*)&addr, sizeof( addr ) ) == SOCKET_ERROR )
        {
            closesocket( listener );
            return;
        }

        int addrlen = sizeof( addr );
        if( getsockname( listener, (struct sockaddr*)&addr, &addrlen ) == SOCKET_ERROR )
        {
            closesocket( listener );
            return;
        }

        if( listen( listener, 1 ) == SOCKET_ERROR )
        {
            closesocket( listener );
            return;
        }

        // Create client socket and connect
        m_writeFd = socket( AF_INET, SOCK_STREAM, IPPROTO_TCP );
        if( m_writeFd == INVALID_SOCKET )
        {
            closesocket( listener );
            return;
        }

        if( connect( m_writeFd, (struct sockaddr*)&addr, sizeof( addr ) ) == SOCKET_ERROR )
        {
            closesocket( m_writeFd );
            closesocket( listener );
            m_writeFd = INVALID_SOCKET;
            return;
        }

        // Accept the connection
        m_readFd = accept( listener, NULL, NULL );
        closesocket( listener );  // Done with listener

        if( m_readFd == INVALID_SOCKET )
        {
            closesocket( m_writeFd );
            m_writeFd = INVALID_SOCKET;
            return;
        }

        // Set non-blocking
        u_long mode = 1;
        ioctlsocket( m_readFd, FIONBIO, &mode );
        ioctlsocket( m_writeFd, FIONBIO, &mode );
    }

    SOCKET m_readFd;
    SOCKET m_writeFd;
#else
    int m_readFd;
    int m_writeFd;
#ifdef __linux__
    int m_eventfd;
#endif
#endif
    std::mutex m_lock;
    bool m_notified = false;
};

}

#endif
