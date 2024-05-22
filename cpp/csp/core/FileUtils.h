#include <csp/core/Exception.h>
#include <filesystem>
#include <string>
#include <sys/stat.h>

namespace csp::utils
{

inline std::string dirname( const std::string & path )
{
    return std::filesystem::path(path).parent_path().string();
}

//files and directories are treated equally
inline bool fileExists( const std::string & fileOrDir )
{
    return std::filesystem::exists(fileOrDir);
}

inline void mkdir( const std::string & path, 
                   std::filesystem::perms perms = std::filesystem::perms::owner_all | std::filesystem::perms::group_all | std::filesystem::perms::others_read | std::filesystem::perms::others_exec  )
{
    if (!fileExists(path))
    {
        std::error_code err;
        if (!std::filesystem::create_directories(path, err))
            CSP_THROW(IOError, "Failed to create path " << path << ": " << err.message());
        std::filesystem::permissions(path, perms);
    }
}

}
