#ifndef _IN_CSP_CORE_EXCEPTION_H
#define _IN_CSP_CORE_EXCEPTION_H

#include <exception>
#include <sstream>
#include <string.h>
#include <csp/core/Likely.h>
#include <csp/core/Platform.h>

namespace csp
{

class Exception : public std::exception
{
public:
    Exception( const char * exType, const std::string & description, const char * file, const char * func, int line ) :
        m_exType( exType ), m_description( description ), m_file( file ), m_function( func ), m_line( line ), m_backtracemessages( nullptr )
    { 
        setbt();
    }

    Exception(const char* exType, const std::string& description) : Exception(exType, description, "", "", -1)
    {}
    ~Exception() { free( m_backtracemessages ); }
    Exception( const Exception & );
    Exception( Exception&& );

    const char * what() const noexcept override { return full( false ).c_str(); }
    const std::string & full( bool includeBacktrace ) const noexcept
    {
        m_full.clear();
        if( m_line >= 0 )
            m_full = m_file + ":" + m_function + ":" + std::to_string( m_line ) + ":";
        m_full += m_exType + ": " + m_description;
        if( includeBacktrace && m_backtracesize > 0 )
            m_full += '\n' + backtraceString();
        return m_full;
    }
    const std::string & description() const noexcept  { return m_description; }

    char ** btmessages() const { return m_backtracemessages; }
    int btsize() const { return m_backtracesize; }

    std::string backtraceString() const;
    void writeBacktrace( std::ostream & ) const; // streams are generally not-copyable (e.g. std::cerr) so we need this overload
    void writeBacktrace( std::ostream && dest ) const { writeBacktrace( dest ); }

private:
    void setbt();

    mutable std::string m_full;
    std::string         m_exType;
    std::string         m_description;
    std::string         m_file;
    std::string         m_function;
    int                 m_line;
    int                 m_backtracesize;
    char **             m_backtracemessages;
};

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define CSP_DECLARE_EXCEPTION( DerivedException, BaseException ) class DerivedException : public BaseException { public: DerivedException( const char * exType, const std::string &r, const char * file, const char * func, int line ) : BaseException( exType, r, file, func, line ) {} };

CSP_DECLARE_EXCEPTION( AssertionError,     Exception )
CSP_DECLARE_EXCEPTION( RuntimeException,   Exception )
CSP_DECLARE_EXCEPTION( InvalidArgument,    RuntimeException )
CSP_DECLARE_EXCEPTION( NotImplemented,     RuntimeException )
CSP_DECLARE_EXCEPTION( ValueError,         RuntimeException )
CSP_DECLARE_EXCEPTION( KeyError,           RuntimeException )
CSP_DECLARE_EXCEPTION( TypeError,          RuntimeException )
CSP_DECLARE_EXCEPTION( RangeError,         RuntimeException )
CSP_DECLARE_EXCEPTION( OverflowError,      RuntimeException )
CSP_DECLARE_EXCEPTION( DivideByZero,       RuntimeException )
CSP_DECLARE_EXCEPTION( RecursionError,     RuntimeException )
CSP_DECLARE_EXCEPTION( IOError,            RuntimeException )
CSP_DECLARE_EXCEPTION( OSError,            RuntimeException )
CSP_DECLARE_EXCEPTION( OutOfMemoryError,   RuntimeException )
CSP_DECLARE_EXCEPTION( FileNotFoundError,  IOError )

template<typename T>
[[noreturn]] NO_INLINE void throw_exc(T&& e);

template<typename T>
[[noreturn]] inline void throw_exc(T&& e) {throw e;}

#define CSP_THROW( EX_TYPE, MSG )         do { std::stringstream desc; desc << MSG ;  csp::throw_exc(EX_TYPE( #EX_TYPE, desc.str(), __FILENAME__ , __FUNCTION__ , __LINE__  )); } while( 0 )
#define CSP_THROW_EX( EX_TYPE, MSG, ... ) do { std::stringstream desc; desc << MSG ;  csp::throw_exc(EX_TYPE( #EX_TYPE, desc.str(), __FILENAME__ , __FUNCTION__ , __LINE__ , __VA_ARGS__ )); } while( 0 )

#define CSP_TRUE_OR_THROW( EXPR, EXCEPTION_TYPE, MESSAGE ) do {if( unlikely(!((EXPR))) ) { CSP_THROW( EXCEPTION_TYPE, MESSAGE ); }} while(false)
#define CSP_TRUE_OR_THROW_RUNTIME(EXPR, MESSAGE) CSP_TRUE_OR_THROW(EXPR, csp::RuntimeException, MESSAGE)
#define CSP_NOT_IMPLEMENTED CSP_THROW(csp::NotImplemented, "");

// This is kind of the same as CSP_ASSERT, the difference is that whatever will be validated with this check, will be checked in both debug and release builds.
// Generally CSP_TRUE_OR_THROW_RUNTIME is better since usually the error message should be customized.
#define CSP_ENSURE_TRUE(EXPR) CSP_TRUE_OR_THROW_RUNTIME(EXPR, #EXPR)

#ifdef	NDEBUG
#define CSP_ASSERT( EXPR ) ( (void ) (0))
#else
#define CSP_ASSERT( EXPR ) CSP_TRUE_OR_THROW(EXPR, csp::AssertionError, #EXPR)
#endif


}

#endif
