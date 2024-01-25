#include <csp/core/Exception.h>
#include <filesystem>
#include <string>
#include <sys/stat.h>

namespace csp::utils
{

inline std::string dirname( const std::string & path )
{
    if( path.length() >= 2 )
        return path.substr( 0, path.rfind( "/", path.length() - 2 ) );
    return "";
}

//files and directories are treated equally
inline bool fileExists( const std::string & fileOrDir )
{
    return std::filesystem::exists(fileOrDir);
}

inline void mkdir( const std::string & path,  mode_t mode = S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH )
{
    size_t pos = 0;
    do
    {
        pos = path.find( '/', pos + 1 );
        std::string subpath = path.substr( 0, pos );
        if( !fileExists( subpath) && ( ::mkdir( subpath.c_str(), mode ) == -1 && errno != EEXIST ) )
            CSP_THROW( IOError, "Failed to create path " << subpath << ": " << strerror( errno ) );
    } while( pos != std::string::npos );
}

}
