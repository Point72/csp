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

inline void mkdir( const std::string & path, 
                   std::filesystem::perms perms = std::filesystem::perms::owner_all | std::filesystem::perms::group_all | std::filesystem::perms::others_read | std::filesystem::perms::others_exec  )
{
    size_t pos = 0;

    do
    {
        pos = path.find( std::filesystem::path::preferred_separator, pos + 1 );
        std::string subpath = path.substr( 0, pos );
        std::error_code err;
        if( !fileExists( subpath ) && ( !std::filesystem::create_directory( subpath, err ) ) )
            CSP_THROW( IOError, "Failed to create path " << subpath << ": " << err.message() );
        std::filesystem::permissions( subpath, perms );
    } while( pos != std::string::npos );
}

}
