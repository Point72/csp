#ifndef _IN_CSP_ADAPTERS_PARQUET_ParquetReaderColumnAdapter_H
#define _IN_CSP_ADAPTERS_PARQUET_ParquetReaderColumnAdapter_H

#include <csp/adapters/utils/StructAdapterInfo.h>
#include <csp/adapters/utils/ValueDispatcher.h>
#include <csp/core/Generator.h>
#include <csp/engine/AdapterManager.h>
#include <csp/engine/CspType.h>
#include <csp/engine/PartialSwitchCspType.h>
#include <csp/engine/Struct.h>
#include <csp/adapters/parquet/DialectGenericListReaderInterface.h>

#include <arrow/table.h>
#include <arrow/type.h>
#include <arrow/array.h>
#include <memory>
#include <optional>
#include <parquet/arrow/reader.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <set>

namespace arrow::io
{
class ReadableFile;
}

namespace csp::adapters::parquet
{
CSP_DECLARE_EXCEPTION( ParquetColumnTypeError, csp::TypeError );

class ParquetReader;
class ParquetStructAdapter;

class ParquetColumnAdapter
{
public:
    ParquetColumnAdapter( ParquetReader &parquetReader, const std::string& columnName ) : m_parquetReader( parquetReader ), m_columnName(columnName){}

    virtual ~ParquetColumnAdapter(){}

    const std::string& getColumnName() const {return m_columnName;}

    virtual bool isListType() const {return false;};
    virtual CspTypePtr  getContainerValueType() const {CSP_THROW(TypeError, "Trying to get list value on non container type");}

    virtual void addSubscriber( ManagedSimInputAdapter *inputAdapter, std::optional<utils::Symbol> symbol = {} ) = 0;
    // NOTE: This API is only defined for ListType Column Adapters
    virtual void addSubscriber( ManagedSimInputAdapter *inputAdapter, std::optional<utils::Symbol> symbol, const DialectGenericListReaderInterface::Ptr &listReader ) = 0;

    virtual void dispatchValue( const utils::Symbol *symbol ) = 0;

    virtual void ensureType( CspType::Ptr cspType ) = 0;

    virtual void readCurValue() = 0;

    ParquetReader &getReader(){ return m_parquetReader; }

    const ParquetReader &getReader() const{ return m_parquetReader; }

    virtual bool isNativeType() const {return getNativeCspType() && getNativeCspType()->type() < CspType::TypeTraits::MAX_NATIVE_TYPE;}
    virtual bool isMissingColumn() const { return false;}
    virtual CspTypePtr getNativeCspType() const =0;

    template< typename T >
    std::optional<T> &getCurValue()
    {
        return *static_cast<std::optional<T> *>(getCurValueUntyped());
    }

    virtual void handleNewBatch( const std::shared_ptr<::arrow::ChunkedArray> &data ) = 0;
    virtual void handleNewBatch( const std::shared_ptr<::arrow::Array> &data ) = 0;

protected:
    friend class ParquetReader;

    virtual void *getCurValueUntyped() = 0;

protected:
    ParquetReader     &m_parquetReader;
    const std::string m_columnName;
};

class ParquetStructAdapter final
{
public:
    using ValueDispatcher = csp::adapters::utils::ValueDispatcher<StructPtr &>;
    using FieldSetter = std::function<void( StructPtr & )>;

    ParquetStructAdapter( ParquetReader &parquetReader,
                          csp::adapters::utils::StructAdapterInfo adapterInfo );
    ParquetStructAdapter( ParquetReader &parquetReader,
                          std::shared_ptr<::arrow::StructType> arrowType,
                          const std::shared_ptr<StructMeta> &structMeta,
                          const std::vector<std::unique_ptr<ParquetColumnAdapter>> &columnAdapters );

    void addSubscriber( ManagedSimInputAdapter *inputAdapter, std::optional<utils::Symbol> symbol = {} );
    void addSubscriber( ValueDispatcher::SubscriberType subscriber, std::optional<utils::Symbol> symbol = {} );

    void dispatchValue( const utils::Symbol *symbol, bool isNull = false );

    const std::shared_ptr<StructMeta> &getStructMeta(){ return m_structMeta; }

    void onSchemaChange(){ m_needsReset = true; }
private:
    void createFieldSetter( const std::string &fieldName, ParquetColumnAdapter &columnAdapter );

private:
    using StructAdapterInfo = csp::adapters::utils::StructAdapterInfo;

    ParquetReader               &m_parquetReader;
    std::shared_ptr<StructMeta> m_structMeta;
    ValueDispatcher             m_valueDispatcher;
    std::vector<FieldSetter>    m_fieldSetters;
    std::function<void()>       m_resetFunc;
    bool                        m_needsReset = false;
};

class MissingColumnAdapter : public ParquetColumnAdapter
{
public:
    using ParquetColumnAdapter::ParquetColumnAdapter;

    virtual void addSubscriber( ManagedSimInputAdapter *inputAdapter, std::optional<utils::Symbol> symbol = {} ) override {};
    virtual void addSubscriber( ManagedSimInputAdapter *inputAdapter, std::optional<utils::Symbol> symbol, const DialectGenericListReaderInterface::Ptr &listReader ) override {};

    virtual void dispatchValue( const utils::Symbol *symbol ) override {};

    virtual void ensureType( CspType::Ptr cspType ) override {};

    virtual void readCurValue() override {};

    bool isNativeType() const override
    {
        CSP_THROW( csp::RuntimeException, "Trying to check type of a missing column " << getColumnName() );
    }

    bool isMissingColumn() const override{ return true; }

    virtual CspTypePtr getNativeCspType() const override
    {
        CSP_THROW( csp::RuntimeException, "Trying to get native type of a missing column " << getColumnName() );
    }

    virtual void handleNewBatch( const std::shared_ptr<::arrow::ChunkedArray> &data ) override
    {
        CSP_THROW( csp::RuntimeException, "Trying to handle new batch for a missing column " << getColumnName() );
    }

    virtual void handleNewBatch( const std::shared_ptr<::arrow::Array> &data ) override
    {
        CSP_THROW( csp::RuntimeException, "Trying to handle new batch for a missing column " << getColumnName() );
    }

protected:
    void *getCurValueUntyped() override
    {
        CSP_THROW( csp::RuntimeException, "Trying to get value of a missing column " << getColumnName() );
    }
};

template< typename ValueType, typename ArrowArrayType, typename ValueDispatcherT = csp::adapters::utils::ValueDispatcher<const ValueType &>>
class BaseTypedColumnAdapter : public ParquetColumnAdapter
{
public:
    BaseTypedColumnAdapter( ParquetReader &parquetReader, const std::string& columnName )
            : ParquetColumnAdapter( parquetReader, columnName )
    {
    }

    void addSubscriber( ManagedSimInputAdapter *inputAdapter, std::optional<utils::Symbol> symbol = {} ) override;
    void addSubscriber( ManagedSimInputAdapter *inputAdapter,
                        std::optional<utils::Symbol> symbol, const DialectGenericListReaderInterface::Ptr &listReader ) override;
    void dispatchValue( const utils::Symbol *symbol ) override;

    void ensureType( CspType::Ptr cspType ) override;

protected:
    void *getCurValueUntyped() override;

    void handleNewBatch( const std::shared_ptr<::arrow::ChunkedArray> &data ) override;
    void handleNewBatch( const std::shared_ptr<::arrow::Array> &data ) override;

protected:
    using CompatibleTypeSwitch = ConstructibleTypeSwitch<ValueType>;
    using ValueDispatcher = ValueDispatcherT;

    ValueDispatcher                 m_dispatcher;
    std::shared_ptr<ArrowArrayType> m_curChunkArray;
    std::optional<ValueType>        m_curValue;
};

template< typename ValueType, typename ArrowArrayType >
class NativeTypeColumnAdapter : public BaseTypedColumnAdapter<ValueType, ArrowArrayType>
{
public:
    using BaseTypedColumnAdapter<ValueType, ArrowArrayType>::BaseTypedColumnAdapter;
    virtual CspTypePtr getNativeCspType() const override {return CspType::fromCType<ValueType>::type();}

protected:
    void readCurValue() override;
};

template< int64_t UNIT >
class DatetimeColumnAdapter : public BaseTypedColumnAdapter<csp::DateTime, arrow::TimestampArray>
{
public:
    using BaseTypedColumnAdapter::BaseTypedColumnAdapter;
    virtual CspTypePtr getNativeCspType() const override {return CspType::fromCType<csp::DateTime>::type();}
protected:
    void readCurValue() override;
};

template< int64_t UNIT >
class DurationColumnAdapter : public BaseTypedColumnAdapter<csp::TimeDelta, arrow::DurationArray>
{
public:
    using BaseTypedColumnAdapter::BaseTypedColumnAdapter;
    virtual CspTypePtr getNativeCspType() const override {return CspType::fromCType<csp::TimeDelta>::type();}
protected:
    void readCurValue() override;
};

template< int64_t UNIT, typename ArrowDateArray >
class DateColumnAdapter : public BaseTypedColumnAdapter<csp::Date, ArrowDateArray>
{
public:
    using BaseTypedColumnAdapter<csp::Date, ArrowDateArray>::BaseTypedColumnAdapter;
    virtual CspTypePtr getNativeCspType() const override {return CspType::fromCType<csp::Date>::type();}
protected:
    void readCurValue() override;
};

template< int64_t UNIT, typename ArrowTimeArray >
class TimeColumnAdapter : public BaseTypedColumnAdapter<csp::Time, ArrowTimeArray>
{
public:
    using BaseTypedColumnAdapter<csp::Time, ArrowTimeArray>::BaseTypedColumnAdapter;
    virtual CspTypePtr getNativeCspType() const override {return CspType::fromCType<csp::Time>::type();}
protected:
    void readCurValue() override;
};

template< typename ArrowStringArrayType >
class StringColumnAdapter : public BaseTypedColumnAdapter<std::string, ArrowStringArrayType>
{
public:
    using BaseTypedColumnAdapter<std::string, ArrowStringArrayType>::BaseTypedColumnAdapter;
    using BaseTypedColumnAdapter<std::string, ArrowStringArrayType>::m_dispatcher;
    void addSubscriber( ManagedSimInputAdapter *inputAdapter, std::optional<utils::Symbol> symbol = {} ) override;
    virtual CspTypePtr getNativeCspType() const override {return nullptr;}
protected:
    void readCurValue() override;
};

template< typename ArrowBytesArrayType >
class BytesColumnAdapter : public BaseTypedColumnAdapter<std::string, ArrowBytesArrayType>
{
public:
    using BaseTypedColumnAdapter<std::string, ArrowBytesArrayType>::BaseTypedColumnAdapter;
    virtual CspTypePtr getNativeCspType() const override {return nullptr;}
protected:
    void readCurValue() override;
};

class FixedSizeBinaryColumnAdapter : public BaseTypedColumnAdapter<std::string, arrow::FixedSizeBinaryArray>
{
public:
    using BaseTypedColumnAdapter::BaseTypedColumnAdapter;
    virtual CspTypePtr getNativeCspType() const override {return nullptr;}
protected:
    void readCurValue() override;
};

class DictionaryColumnAdapter : public BaseTypedColumnAdapter<std::string, arrow::DictionaryArray>
{
public:
    using BaseTypedColumnAdapter::BaseTypedColumnAdapter;
    virtual CspTypePtr getNativeCspType() const override {return nullptr;}
protected:
    void readCurValue() override;
};

template< typename ArrowListArrayType, typename ValueArrayType, typename ValueType>
class BaseListColumnAdapter : public BaseTypedColumnAdapter<DialectGenericType, ArrowListArrayType>
{
public:
    using BaseTypedColumnAdapter<DialectGenericType, ArrowListArrayType>::BaseTypedColumnAdapter;
    using BaseTypedColumnAdapter<DialectGenericType, ArrowListArrayType>::getColumnName;
    void addSubscriber( ManagedSimInputAdapter *inputAdapter, std::optional<utils::Symbol> symbol = {} ) override;
    void addSubscriber( ManagedSimInputAdapter *inputAdapter,
                        std::optional<utils::Symbol> symbol, const DialectGenericListReaderInterface::Ptr &listReader ) override;
    CspTypePtr getNativeCspType() const override {return nullptr;}
    bool isListType() const override{ return true; };
    CspTypePtr getContainerValueType() const override{ return CspType::fromCType<ValueType>::type(); }
protected:
    // For now we allow only one subscription for list columns. In the future we might allow multiple subscriptions using different readers.
    // For example we could subscritbe to the same column as a list and as array, we would need to do more book keepting in this case.
    typename TypedDialectGenericListReaderInterface<ValueType>::Ptr m_listReader = nullptr;
    using BaseTypedColumnAdapter<DialectGenericType, ArrowListArrayType>::m_parquetReader;
};

template< typename ArrowListArrayType, typename ValueArrayType, typename ValueType=typename ValueArrayType::TypeClass::c_type >
class NativeListColumnAdapter : public BaseListColumnAdapter<ArrowListArrayType, ValueArrayType, ValueType>
{
public:
    using BaseListColumnAdapter<ArrowListArrayType, ValueArrayType, ValueType>::BaseListColumnAdapter;
    using BaseListColumnAdapter<ArrowListArrayType, ValueArrayType, ValueType>::addSubscriber;
protected:
    using BaseListColumnAdapter<ArrowListArrayType, ValueArrayType, ValueType>::m_listReader;
    void readCurValue() override;
};

template< typename ArrowListArrayType, typename ArrowBytesArrayType >
class BytesListColumnAdapter: public BaseListColumnAdapter<ArrowListArrayType, ArrowBytesArrayType, std::string>
{
public:
    using BaseListColumnAdapter<ArrowListArrayType, ArrowBytesArrayType, std::string>::BaseListColumnAdapter;
    using BaseListColumnAdapter<ArrowListArrayType, ArrowBytesArrayType, std::string>::addSubscriber;
protected:
    using BaseListColumnAdapter<ArrowListArrayType, ArrowBytesArrayType, std::string>::m_listReader;
    void readCurValue() override;
};

class StructColumnAdapter : public BaseTypedColumnAdapter<StructPtr, arrow::StructArray, csp::adapters::utils::ValueDispatcher<StructPtr &>>
{
public:
    using BASE = BaseTypedColumnAdapter<StructPtr, arrow::StructArray, csp::adapters::utils::ValueDispatcher<StructPtr &>>;

    StructColumnAdapter( ParquetReader &parquetReader, const std::shared_ptr<::arrow::StructType> &arrowType,
                         const std::string& columnName )
        : BASE( parquetReader, columnName ), m_arrowType( arrowType )
    {
    }

    virtual CspTypePtr getNativeCspType() const override {return nullptr;}

    void setStructMeta( const std::shared_ptr<StructMeta> &structMeta ){ initFromStructMeta( structMeta ); }
    void addSubscriber( ManagedSimInputAdapter *inputAdapter, std::optional<utils::Symbol> symbol = {} ) override;
protected:
    void readCurValue() override;
    void handleNewBatch( const std::shared_ptr<::arrow::ChunkedArray> &data ) override;
    void handleNewBatch( const std::shared_ptr<::arrow::Array> &data ) override;
private:
    void initFromStructMeta( const std::shared_ptr<StructMeta> &structMeta );

private:
    friend class ParquetStructAdapter;

    std::shared_ptr<::arrow::StructType>               m_arrowType;
    std::unique_ptr<ParquetStructAdapter>              m_structAdapter;
    std::vector<std::unique_ptr<ParquetColumnAdapter>> m_childColumnAdapters;
};

std::unique_ptr<ParquetColumnAdapter> createColumnAdapter(
        ParquetReader &parquetReader,
        const ::arrow::Field& field,
        const std::string& fileName,
        const std::map<std::string, std::shared_ptr<StructMeta>>* structMetaByColumnName = nullptr );

std::unique_ptr<ParquetColumnAdapter> createMissingColumnAdapter(ParquetReader &parquetReader,const std::string& columnName);

}


#endif
