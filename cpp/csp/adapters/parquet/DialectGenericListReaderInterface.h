#ifndef _IN_CSP_ADAPTERS_PARQUET_DialectGenericListReaderInterface_H
#define _IN_CSP_ADAPTERS_PARQUET_DialectGenericListReaderInterface_H

#include <memory>
#include <csp/engine/DialectGenericType.h>
#include <csp/adapters/parquet/ParquetReaderColumnAdapter.h>
#include <arrow/array/builder_base.h>

namespace csp::adapters::parquet
{

class DialectGenericListReaderInterface
{
public:
    using Ptr = std::shared_ptr<DialectGenericListReaderInterface>;
    virtual ~DialectGenericListReaderInterface() = default;
    virtual csp::DialectGenericType create( uint32_t size ) = 0;
    /**
     * This form of creation is needed for elements that need to know the element size in advance, such as numpy arrays of strings
     * @param size
     * @return
     */
    virtual csp::DialectGenericType create( uint32_t size, uint32_t maxElementSize ) = 0;
    virtual CspTypePtr getValueType() = 0;
};

template< typename T >
class TypedDialectGenericListReaderInterface : public DialectGenericListReaderInterface
{
public:
    using Ptr = std::shared_ptr<TypedDialectGenericListReaderInterface<T>>;
    CspTypePtr getValueType() override{ return CspType::fromCType<T>::type(); }

    /**
     * Return a raw data buffer for the object that was generated using the call to create. Should return the pointer to internal buffer
     * that can be written directly into if that's supported, otherwise, should return nullptr in which case setValue will be used for
     * inserting the elements
     * @param list: The list object that was created using the call to create function
     * @return Internal write buffer of the list of nullptr if the list doesn't support writing to internal buffer
     */
    virtual T *getRawDataBuffer( const csp::DialectGenericType &list ) const{ return nullptr; }
    virtual void setValue( const csp::DialectGenericType &list, int index, const T &value ) = 0;
};


}

#endif