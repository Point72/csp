#ifndef _IN_CSP_ADAPTERS_PARQUET_FileNameGeneratorReplicator_H
#define _IN_CSP_ADAPTERS_PARQUET_FileNameGeneratorReplicator_H

#include <csp/core/Generator.h>
#include <csp/core/Time.h>

#include <memory>
#include <string>
#include <vector>

namespace csp::adapters::parquet
{

/**
 * Consumes a string generator (producing folder/file names) and caches the results so they can be
 * replayed by multiple child generators.  Each replica can optionally append a suffix.
 */
class FileNameGeneratorReplicator
{
public:
    using Ptr = std::shared_ptr<FileNameGeneratorReplicator>;
    using GeneratorPtr = csp::Generator<std::string, csp::DateTime, csp::DateTime>::Ptr;

    FileNameGeneratorReplicator( GeneratorPtr source )
            : m_generatorPtr( source )
    {
    }

    void init( csp::DateTime start, csp::DateTime end )
    {
        m_generatorPtr -> init( start, end );
    }

    const std::vector<std::string> &getFileNames(){ return m_fileNames; }

    void consumeNextGeneratorFile()
    {
        std::string nextFile;
        if( m_generatorPtr -> next( nextFile ) )
        {
            m_fileNames.push_back( std::move( nextFile ) );
        }
    }

    GeneratorPtr getGeneratorReplica( const std::string &suffix = "" )
    {
        return std::make_shared<ChildGenerator>( *this, suffix );
    }

private:
    class ChildGenerator : public csp::Generator<std::string, csp::DateTime, csp::DateTime>
    {
    public:
        ChildGenerator( FileNameGeneratorReplicator &owner, const std::string &suffix )
                : m_owner( owner ), m_suffix( suffix ), m_nextIndex( 0 )
        {
        }

        void init( csp::DateTime, csp::DateTime ){}

        bool next( std::string &value )
        {
            if( m_nextIndex < 0 )
            {
                return false;
            }
            const std::vector<std::string> &folders = m_owner.getFileNames();
            if( m_nextIndex >= static_cast<int>(folders.size()) )
            {
                m_owner.consumeNextGeneratorFile();
            }
            if( m_nextIndex >= static_cast<int>(folders.size() ) )
            {
                m_nextIndex = -1;
                return false;
            }

            value = folders[ m_nextIndex++ ] + m_suffix;
            return true;
        }

    private:
        FileNameGeneratorReplicator &m_owner;
        const std::string           m_suffix;
        int                         m_nextIndex = 0;
    };

private:
    GeneratorPtr             m_generatorPtr;
    std::vector<std::string> m_fileNames;
};

} // namespace csp::adapters::parquet

#endif
