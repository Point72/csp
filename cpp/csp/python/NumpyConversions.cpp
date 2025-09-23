//this is included first so that we do include without NO_IMPORT_ARRAY defined, which is done in NumpyConversions.h
#include <numpy/ndarrayobject.h>
#include <numpy/npy_2_compat.h>


#include <csp/core/Time.h>
#include <csp/python/NumpyConversions.h>

#include <locale>
#include <codecvt>

static void * init_nparray()
{
    csp::python::AcquireGIL gil;
    import_array();
    return nullptr;
}
static void * s_init_array = init_nparray();

namespace csp::python
{

PyObject * valuesAtIndexToNumpy( ValueType valueType, const csp::TimeSeriesProvider * ts, int32_t startIndex, int32_t endIndex,
                                 autogen::TimeIndexPolicy startPolicy, autogen::TimeIndexPolicy endPolicy,
                                 DateTime startDt, DateTime endDt )
{
    return switchCspType( ts -> type(),
                          [ valueType, ts, startIndex, endIndex, startPolicy, endPolicy, startDt, endDt ]( auto tag )
                          {
                              return createNumpyArray<typename decltype(tag)::type>( valueType, ts, startIndex, endIndex, startPolicy,
                                                                                     endPolicy, startDt, endDt );
                          } );
}

int64_t scalingFromNumpyDtUnit( NPY_DATETIMEUNIT base )
{
    switch( base )
    {
        // https://github.com/numpy/numpy/blob/v2.0.0/numpy/__init__.pxd#L794
        case NPY_FR_Y:
            return csp::TimeDelta::fromDays(365).asNanoseconds();
        case NPY_FR_M:
            return csp::TimeDelta::fromDays(30).asNanoseconds();
        case NPY_FR_W:
            return csp::TimeDelta::fromDays(7).asNanoseconds();
#ifdef NPY_FR_B
        case NPY_FR_B:
            return csp::TimeDelta::fromDays(5).asNanoseconds();
#endif
        case NPY_FR_D:
            return csp::TimeDelta::fromDays(1).asNanoseconds();
        case NPY_FR_h:
            return csp::TimeDelta::fromHours(1).asNanoseconds();
        case NPY_FR_m:
            return csp::TimeDelta::fromMinutes(1).asNanoseconds();
        case NPY_FR_s:
            return csp::TimeDelta::fromSeconds(1).asNanoseconds();
        case NPY_FR_ms:
            return csp::TimeDelta::fromMilliseconds(1).asNanoseconds();
        case NPY_FR_us:
            return csp::TimeDelta::fromMicroseconds(1).asNanoseconds();
        case NPY_FR_ns:
            return 1;
        // unsupported or invalid units
        // enumerated here for clarity in error messages
        case NPY_FR_ps:
            CSP_THROW(csp::NotImplemented, "datetime resolution not supported or invalid - saw NPY_DATETIMEUNIT value NPY_FR_ps" );
            return 0;
        case NPY_FR_fs:
            CSP_THROW(csp::NotImplemented, "datetime resolution not supported or invalid - saw NPY_DATETIMEUNIT value NPY_FR_fs" );
            return 0;
        case NPY_FR_GENERIC:
            CSP_THROW(csp::NotImplemented, "datetime resolution not supported or invalid - saw NPY_DATETIMEUNIT value NPY_FR_generic" );
            return 0;
        case NPY_FR_ERROR:
            CSP_THROW(csp::NotImplemented, "datetime resolution not supported or invalid - saw NPY_DATETIMEUNIT value NPY_FR_error" );
            return 0;
        default:
            if(static_cast<int>(base) == 3) {
                // NPY_FR_B was removed in numpy 1.20
                return csp::TimeDelta::fromDays(5).asNanoseconds();
            }
            CSP_THROW(csp::NotImplemented, "datetime resolution not supported or invalid - saw NPY_DATETIMEUNIT value " << static_cast<int32_t>(base) );
            return 0;
    }
}

NPY_DATETIMEUNIT datetimeUnitFromDescr( PyArray_Descr* descr )
{
    PyArray_DatetimeDTypeMetaData* dtypeMeta;
    if (PyArray_RUNTIME_VERSION >= NPY_2_0_API_VERSION)
    {
        // NumPy 2.x way
        dtypeMeta = ( PyArray_DatetimeDTypeMetaData * ) ( ( (_PyArray_LegacyDescr *) descr ) -> c_metadata );
    }
    else
    {
        // NumPy 1.x way
        dtypeMeta = ( PyArray_DatetimeDTypeMetaData * ) ( ( (PyArray_DescrProto *) descr ) -> c_metadata );
    }
    PyArray_DatetimeMetaData* dtMeta = &( dtypeMeta -> meta );
    return dtMeta -> base;
}

static std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> wstr_converter;

void stringFromNumpyStr( void* data, std::string& out, char numpy_type, int elem_size_bytes )
{
    // strings from numpy arrays are fixed width and zero filled.
    // if the last char is 0, can treat as null terminated, else use full width

    if( numpy_type == NPY_UNICODELTR)
    {
        const char32_t* const raw_value = (const char32_t *) data;
        const int field_size = elem_size_bytes / sizeof( char32_t );

        if( raw_value[field_size - 1] == 0 )
        {
            std::u32string wstr( raw_value );
            out = wstr_converter.to_bytes( wstr );
        }
        else
        {
            std::u32string wstr( raw_value, field_size );
            out = wstr_converter.to_bytes( wstr );
        }
    }
    else if( numpy_type == NPY_STRINGLTR )
    {
        const char * const raw_value = (const char *) data;

        if( raw_value[elem_size_bytes - 1] == 0 )
            out = std::string( raw_value );
        else
            out = std::string( raw_value, elem_size_bytes );
    }
    else if( numpy_type == NPY_CHARLTR )
    {
        const char * const raw_value = (const char *) data;
        out = std::string( raw_value, 1 );
    }
}

void validateNumpyTypeVsCspType( const CspTypePtr & type, char numpy_type_char )
{
    csp::CspType::Type cspType = type -> type();

    switch( numpy_type_char )
    {
        case NPY_BOOLLTR:
            if( cspType != csp::CspType::Type::BOOL )
                CSP_THROW( ValueError, "numpy type " << numpy_type_char << " requires bool output type" );
            break;
        case NPY_BYTELTR:
        case NPY_UBYTELTR:
        case NPY_SHORTLTR:
        case NPY_USHORTLTR:
        case NPY_INTLTR:
        case NPY_UINTLTR:
        case NPY_LONGLTR:
        case NPY_LONGLONGLTR:
            if( cspType != csp::CspType::Type::INT64 )
                CSP_THROW( ValueError, "numpy type " << numpy_type_char << " requires int output type" );
            break;
        case NPY_ULONGLTR:
        case NPY_ULONGLONGLTR:
            CSP_THROW( ValueError, "numpy type " << numpy_type_char << " (int type that can't cleanly convert to int64) not supported" );
        case NPY_HALFLTR:
            CSP_THROW( ValueError, "numpy type " << numpy_type_char << " (numpy half float) not supported" );
        case NPY_FLOATLTR:
        case NPY_DOUBLELTR:
            if( cspType != csp::CspType::Type::DOUBLE )
                CSP_THROW( ValueError, "numpy type " << numpy_type_char << " requires float output type" );
            break;
        case NPY_LONGDOUBLELTR:
            CSP_THROW( ValueError, "numpy type " << numpy_type_char << " (long double) not supported" );
        case NPY_CFLOATLTR:
        case NPY_CDOUBLELTR:
        case NPY_CLONGDOUBLELTR:
            CSP_THROW( ValueError, "numpy complex type only supported with dtype='object'" );
        case NPY_OBJECTLTR:
            // everything works as object
            break;
        case NPY_STRINGLTR:
        case NPY_UNICODELTR:
        case NPY_CHARLTR:
            if( cspType != csp::CspType::Type::STRING )
                CSP_THROW( ValueError, "numpy type " << numpy_type_char << " requires string output type" );
            break;
        case NPY_VOIDLTR:
            CSP_THROW( ValueError, "numpy void type not supported" );
        case NPY_DATETIMELTR:
            if( cspType != csp::CspType::Type::DATETIME )
                CSP_THROW( ValueError, "numpy type " << numpy_type_char << " requires datetime output type" );
            break;
        case NPY_TIMEDELTALTR:
            if( cspType != csp::CspType::Type::TIMEDELTA )
                CSP_THROW( ValueError, "numpy type " << numpy_type_char << " requires timedelta output type" );
            break;
        default:
            CSP_THROW( ValueError, "unrecognized numpy type:" << numpy_type_char );
    }
}


}
