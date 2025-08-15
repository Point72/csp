#include <csp/core/Exception.h>
#include <csp/core/System.h>

#include <signal.h>
#include <string.h>

#include <iostream>
#include <cstdlib>
#include <stdexcept>
#include <memory>

#ifndef WIN32
#include <execinfo.h>
#endif

//From https://stackoverflow.com/questions/2443135/how-do-i-find-where-an-exception-was-thrown-in-c/2443366
static void csp_terminate( void );
static bool set_sigabrt_handler();

// invoke set_terminate as part of global constant initialization
// This for uncaught exceptions
static const bool SET_TERMINATE = std::set_terminate( csp_terminate );
static const bool SET_SIGABRT_HANDLER = set_sigabrt_handler();

static void printBacktrace( char ** messages, int size, std::ostream & dest )
{
    if( !messages )
    {
        dest << "Backtrace Failed...\n" << std::endl;
        return;
    }

    for( int i = 0; i < size; ++i )
    {
        char *begin_name = 0, *begin_offset = 0;
        char tmp[1024];
        strncpy( tmp, messages[i], sizeof(tmp) - 1 );
        tmp[ sizeof( tmp ) - 1 ] = 0;

        // find parentheses and +address offset surrounding the mangled name:
        // ./module(function+0x15c) [0x8048a6d]
        for (char *p = tmp; *p; ++p)
        {
            if (*p == '(')
                begin_name = p;
            else if (*p == '+')
            {
                begin_offset = p;
                break;
            }
        }

        if (begin_name && begin_offset
            && begin_name < begin_offset)
        {
            begin_name++;
            *begin_offset = '\0';

#ifndef WIN32
            int status;
            char* demangled = abi::__cxa_demangle(begin_name, NULL, NULL, &status);
            dest << "[bt]: (" << i << ") " << ( status == 0 ? demangled : messages[i] ) << std::endl;
            free( demangled );
#else
            dest << "[bt]: (" << i << ") " << messages[i] << std::endl;
#endif
        }
        else
        {
            dest << "[bt]: (" << i << ") " <<  messages[i] << std::endl;
        }
    }

    dest << std::endl;
}

void printBacktrace()
{
//TODO get stack traces on windows
#ifndef WIN32
    void *array[50];
    int size = backtrace( array, 50 );
    auto messages = backtrace_symbols( array, size );
    printBacktrace( messages, size, std::cerr );
    free( messages );
#endif
}

void csp_terminate()
{
    static int tried_throw = 0;

    try 
    {
        // try once to re-throw currently active exception
        if( !tried_throw++ ) 
            throw;
    }
    catch( const csp::Exception & ex )
    {
        std::cerr << __FUNCTION__ << " caught unhandled csp::Exception. what(): "
                  << ex.what() << std::endl;
        if( ex.btsize() > 0 )
            printBacktrace( ex.btmessages(), ex.btsize(), std::cerr );
    }
    catch( const std::exception & e )
    {
        std::cerr << __FUNCTION__ << " caught unhandled std::exception. what(): "
                  << e.what() << std::endl;
    }
    catch( ... ) 
    {
        std::cerr << __FUNCTION__ << " caught unknown/unhandled exception." 
                  << std::endl;
    }

    printBacktrace();

    signal( SIGABRT, SIG_DFL );
    signal( SIGSEGV, SIG_DFL );

    abort();
}

//This is for coredumps
#ifndef WIN32
void sigabrt_handler(int sig_num, siginfo_t* info, void* ctx)
{
    std::cerr << "signal " << sig_num
        << " (" << strsignal(sig_num) << "), address is "
        << info -> si_addr << " from " << std::endl;

    printBacktrace();

    signal(SIGABRT, SIG_DFL);
    signal(SIGSEGV, SIG_DFL);
    signal(SIGBUS, SIG_DFL);

    abort();
}

bool set_sigabrt_handler()
{
    static struct sigaction sigact;

    sigact.sa_sigaction = sigabrt_handler;
    sigact.sa_flags = SA_RESTART | SA_SIGINFO;

    sigaction(SIGABRT, &sigact, NULL);
    sigaction(SIGSEGV, &sigact, NULL);
    sigaction(SIGBUS, &sigact, NULL);
    return true;
}
#else
void sigabrt_handler(int sig_num)
{
    std::cerr << "signal " << sig_num << " from " << std::endl;
    printBacktrace();

    signal(SIGABRT, SIG_DFL);
    signal(SIGSEGV, SIG_DFL);
    abort();
}

bool set_sigabrt_handler()
{
    signal(SIGABRT, sigabrt_handler);
    signal(SIGSEGV, sigabrt_handler);
    return true;
}
#endif

void csp::Exception::setbt()
{
#ifndef WIN32
    void *array[50];
    m_backtracesize = backtrace( array, 50 );
    char **messages = backtrace_symbols( array, m_backtracesize );
    m_backtracemessages = messages;
#endif
}

static char ** dupe_backtraces( char** bt, int n )
{
    if( bt == nullptr ) return nullptr;

    size_t len = n * sizeof( char * );
    for( int i = 0; i < n; ++i )
        len += strlen( bt[ i ] ) + 1;
    char ** newbt = (char **) malloc( len );
    memcpy( newbt, bt, len );

    for( int i = 0; i < n; ++i )
        newbt[i] = (char *)( newbt + ( ( (char **) bt[ i ] ) - bt ) );

    return newbt;
}

csp::Exception::Exception( const csp::Exception &orig ) :
    m_full( orig.m_full ),
    m_exType( orig.m_exType ),
    m_description( orig.m_description ),
    m_file( orig.m_file ),
    m_function( orig.m_function ),
    m_line( orig.m_line ),
    m_backtracesize( orig.m_backtracesize ),
    m_backtracemessages( dupe_backtraces( orig.m_backtracemessages, orig.m_backtracesize ) )
{
}

csp::Exception::Exception( csp::Exception && donor ) :
    m_full( std::move( donor.m_full ) ),
    m_exType( std::move( donor.m_exType ) ),
    m_description( std::move( donor.m_description ) ),
    m_file( std::move( donor.m_file ) ),
    m_function( std::move( donor.m_function ) ),
    m_line( donor.m_line ),
    m_backtracesize( donor.m_backtracesize ),
    m_backtracemessages( donor.m_backtracemessages )
{
    donor.m_backtracemessages = nullptr;
}

void csp::Exception::writeBacktrace( std::ostream & dest ) const
{
    if( m_backtracesize != 0 )
        printBacktrace( m_backtracemessages, m_backtracesize, dest );
}

std::string csp::Exception::backtraceString() const
{
    std::stringstream out;
    writeBacktrace( out );
    return out.str();
}
