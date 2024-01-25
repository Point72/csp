#include <csp/engine/Node.h>
#include <csp/python/Conversions.h>
#include <csp/python/Exception.h>
#include <csp/python/InitHelper.h>
#include <csp/python/PyInputProxy.h>
#include <csp/python/PyNode.h>
#include <csp/python/PyConstants.h>
#include <csp/core/Exception.h>


namespace csp::python
{

PyInputProxy::PyInputProxy( PyNode * node, InputId id ) : m_node( node ),
                                                          m_id( id )
{
}

PyInputProxy * PyInputProxy::create( PyNode * node, InputId id )
{
    PyInputProxy * proxy = ( PyInputProxy * ) PyType.tp_new( &PyType, nullptr, nullptr );
    new( proxy ) PyInputProxy( node, id );
    return proxy;
}

TimeSeriesProvider * PyInputProxy::ts() const
{
    return const_cast<TimeSeriesProvider * >( m_node -> input( m_id ) );
}
 
bool PyInputProxy::ticked() const
{
    return m_node -> inputTicked( m_id );
}

bool PyInputProxy::valid() const
{
    return ts() -> valid();
}

uint32_t PyInputProxy::count() const
{
    return ts() -> count();
}

uint32_t PyInputProxy::num_ticks() const
{
    return ts() -> numTicks();
}

bool PyInputProxy::makeActive()
{
    return m_node -> makeActive( m_id );
}

bool PyInputProxy::makePassive()
{
    return m_node -> makePassive( m_id );
}

Scheduler::Handle PyInputProxy::scheduleAlarm( DateTimeOrTimeDelta dtd, PyObject * value )
{
    //TODO assert is alarm
    auto * alarm = static_cast<AlarmInputAdapter<PyObjectPtr> *>( const_cast<TimeSeriesProvider *>( ts() ) );

    //as noted in PyNode, we always create alarms in PyNode as PyObjectPtr
    if( dtd.isDateTime() )
        return alarm -> scheduleAlarm( dtd.datetime(), PyObjectPtr::incref( value ) );
    else
        return alarm -> scheduleAlarm( dtd.timedelta(), PyObjectPtr::incref( value ) );
}

Scheduler::Handle PyInputProxy::rescheduleAlarm( Scheduler::Handle handle, DateTimeOrTimeDelta dtd )
{
    auto * alarm = static_cast<AlarmInputAdapter<PyObjectPtr> *>( const_cast<TimeSeriesProvider *>( ts() ) );

    if( dtd.isDateTime() )
        return alarm -> rescheduleAlarm( handle, dtd.datetime() );
    else
        return alarm -> rescheduleAlarm( handle, dtd.timedelta() );
}

void PyInputProxy::cancelAlarm( Scheduler::Handle handle )
{
    auto * alarm = static_cast<AlarmInputAdapter<PyObjectPtr> *>( const_cast<TimeSeriesProvider *>( ts() ) );

    alarm -> cancelAlarm( handle );
}

void PyInputProxy::setBufferingPolicy( int32_t tickCount, TimeDelta tickHistory )
{
    if( tickCount > 0 )
        ts() -> setTickCountPolicy( tickCount );
 
    if( !tickHistory.isNone() && tickHistory > TimeDelta::ZERO() )
        ts() -> setTickTimeWindowPolicy( tickHistory );
}

PyObject *PyInputProxy::valueAt( ValueType valueType, PyObject *indexArg, PyObject *duplicatePolicyArg, PyObject *defaultValueArg ) const
{
    int32_t valueIndex;

    if( PyLong_Check( indexArg ) )
    {
        valueIndex = fromPython<int32_t>(indexArg);
        CSP_TRUE_OR_THROW_RUNTIME( valueIndex <= 0,
                                   "Expected non positive value for value_at index, got " << valueIndex );
        // Convert negative value to actual positive index
        valueIndex = -valueIndex;

        if( unlikely( valueIndex >= ts() -> tickCountPolicy() ) )
            CSP_THROW( RangeError, "buffer index out of range.  requesting data at index " << valueIndex << " with buffer policy set to " << 
                       ts() -> tickCountPolicy() << " ticks in node '" << m_node -> name() << "'" );

    }
    else 
    {
        auto dtd = fromPython<DateTimeOrTimeDelta>( indexArg );
        auto duplicatePolicyIntVal = fromPython<int32_t>( duplicatePolicyArg );
        CSP_TRUE_OR_THROW_RUNTIME( duplicatePolicyIntVal == TimeSeriesProvider::DuplicatePolicyEnum::LAST_VALUE ||
                                   duplicatePolicyIntVal == TimeSeriesProvider::DuplicatePolicyEnum::FIRST_VALUE,
                                   "Unsupported duplicate policy " << duplicatePolicyIntVal );

        if( dtd.isTimeDelta() ) 
        {
            auto timedelta = dtd.timedelta();
            CSP_TRUE_OR_THROW_RUNTIME( !timedelta.isNone(),
                                       "None time delta is unsupported" );
            CSP_TRUE_OR_THROW_RUNTIME( timedelta.sign() <= 0,
                                       "Positive time delta is unsupported" );

            if( unlikely( -timedelta > ts() -> tickTimeWindowPolicy() ) )
                CSP_THROW( RangeError, "buffer timedelta out of range.  requesting data at timedelta " << PyObjectPtr::incref( indexArg ) << " with buffer policy set to " << 
                           PyObjectPtr::own( toPython( ts() -> tickTimeWindowPolicy() ) ) << " in node '" << m_node -> name() << "'" );
            valueIndex = ts() -> getValueIndex( m_node -> now() + timedelta, duplicatePolicyIntVal );
        }
        else 
        {
            CSP_TRUE_OR_THROW_RUNTIME( dtd.isDateTime(), "value_at index must be integer, DateTime, or TimeDelta" );
            auto datetime = dtd.datetime();

            CSP_TRUE_OR_THROW_RUNTIME( datetime <= m_node -> now(),
                                       "requesting data from future time" );

            if( unlikely( m_node -> now() - datetime >= ts() -> tickTimeWindowPolicy() ) )
                CSP_THROW( RangeError, "requested buffer time out of range.  requesting datetime " << PyObjectPtr::incref( indexArg ) << " at time " << 
                           PyObjectPtr::own( toPython( m_node -> now() ) ) << " with buffer time window policy set to " << 
                           PyObjectPtr::own( toPython( ts() -> tickTimeWindowPolicy() ) )<< " in node '" << m_node -> name() << "'" );

            valueIndex = ts() -> getValueIndex( datetime, duplicatePolicyIntVal );
        }
    }

    if( valueIndex < 0 || static_cast<uint32_t>(valueIndex) >= num_ticks() )
    {
        if( defaultValueArg == constants::UNSET() )
            CSP_THROW( OverflowError, "No matching value found" );

        return toPython( defaultValueArg );
    }

    switch( valueType )
    {
        case ValueType::VALUE:
            return valueAtIndexToPython( ts(), valueIndex );
        case ValueType::TIMESTAMP:
            return toPython( ts() -> timeAtIndex( valueIndex ) );
        case ValueType::TIMESTAMP_VALUE_TUPLE:
            return PyTuple_Pack( 2, toPython( ts() -> timeAtIndex( valueIndex ) ), valueAtIndexToPython( ts(), valueIndex ) );
        default:
            CSP_THROW( NotImplemented, "Unsupported value type " << valueType );
    }
}

int32_t PyInputProxy::computeStartIndex( DateTime startDt, autogen::TimeIndexPolicy startPolicy ) const
{
    int32_t startIndex;

    switch( startPolicy.enum_value() )
    {
        case autogen::TimeIndexPolicy::enum_::INCLUSIVE:
            startIndex = ts() -> getValueIndex( startDt, TimeSeries::DuplicatePolicyEnum::FIRST_VALUE );
            if( startIndex != -1 && ts() -> timeAtIndex( startIndex ) < startDt )
            {
                startIndex -= 1;
                if( startIndex == -1 || ts() -> timeAtIndex( startIndex ) < startDt )
                    return -1;
            }
            break;
        case autogen::TimeIndexPolicy::enum_::EXCLUSIVE:
            startIndex = ts() -> getValueIndex( startDt, TimeSeries::DuplicatePolicyEnum::LAST_VALUE );
            //if the timestamp matches, we need to exclude it
            if( startIndex != -1 && ts() -> timeAtIndex( startIndex ) <= startDt )
            {
                startIndex -= 1;
                if( startIndex == -1 || ts() -> timeAtIndex( startIndex ) <= startDt )
                    return -1;
            }
            break;
        case autogen::TimeIndexPolicy::enum_::EXTRAPOLATE:
            startIndex = ts() -> getValueIndex( startDt, TimeSeries::DuplicatePolicyEnum::LAST_VALUE );
            break;
        default:
            CSP_THROW( InvalidArgument, "Unsupported time index policy " << startPolicy.name() );
    }

    if( startIndex == -1 )
        startIndex = ts() -> numTicks() - 1;

    return startIndex;
}

int32_t PyInputProxy::computeEndIndex( DateTime endDt, autogen::TimeIndexPolicy endPolicy ) const
{
    int32_t endIndex;

    switch( endPolicy.enum_value() )
    {
        // for end index, inclusive is the same as forced
        case autogen::TimeIndexPolicy::enum_::INCLUSIVE:
        case autogen::TimeIndexPolicy::enum_::EXTRAPOLATE:
            endIndex = ts() -> getValueIndex( endDt, TimeSeries::DuplicatePolicyEnum::LAST_VALUE );
            break;
        case autogen::TimeIndexPolicy::enum_::EXCLUSIVE:
            endIndex = ts() -> getValueIndex( endDt, TimeSeries::DuplicatePolicyEnum::FIRST_VALUE );
            if( endIndex != -1 && ts() -> timeAtIndex( endIndex ) == endDt )
                endIndex += 1;
            break;
        default:
            CSP_THROW( InvalidArgument, "Unsupported time index policy " << endPolicy.name() );
    }

    return endIndex;
}

PyObject *PyInputProxy::valuesAt( ValueType valueType, PyObject *startIndexArg,
                                  PyObject *endIndexArg,
                                  PyObject *startIndexPolicyArg,
                                  PyObject *endIndexPolicyArg ) const
{
    int32_t startIndex, endIndex;
    auto startPolicy = static_cast<autogen::TimeIndexPolicy>( static_cast<PyCspEnum *>( startIndexPolicyArg ) -> enum_ );
    auto endPolicy = static_cast<autogen::TimeIndexPolicy>( static_cast<PyCspEnum *>( endIndexPolicyArg ) -> enum_ );

    if( startIndexArg == Py_None )
        startIndex = 1 - ts() -> numTicks();

    if( endIndexArg == Py_None )
        endIndex = 0;

    if( PyLong_Check( startIndexArg ) || startIndexArg == Py_None )
    {
        CSP_TRUE_OR_THROW_RUNTIME( PyLong_Check( endIndexArg ) || endIndexArg == Py_None,
                                   "End index must be same type as start index" );

        if( startIndexArg != Py_None )
            startIndex = fromPython<int32_t>( startIndexArg );

        if( endIndexArg != Py_None )
            endIndex = fromPython<int32_t>( endIndexArg );

        CSP_TRUE_OR_THROW_RUNTIME( startIndex <= 0,
                                   "Expected non positive value for values_at start index, got " << startIndex );
        CSP_TRUE_OR_THROW_RUNTIME( endIndex <= 0,
                                   "Expected non positive value for values_at end index, got " << endIndex );
        CSP_TRUE_OR_THROW_RUNTIME( endIndex >= startIndex,
                                   "Start index (got) " << startIndex << " must come before end index (got) " << endIndex );
        // Convert negative value to actual positive index
        startIndex = -startIndex;
        endIndex   = -endIndex;

        if( startPolicy.enum_value() != autogen::TimeIndexPolicy::enum_::INCLUSIVE )
            CSP_THROW( InvalidArgument, "Unsupported time index policy for integer indexing: " << startPolicy.name() );

        if( endPolicy.enum_value() != autogen::TimeIndexPolicy::enum_::INCLUSIVE )
            CSP_THROW( InvalidArgument, "Unsupported time index policy for integer indexing: " << endPolicy.name() );

        if( unlikely( startIndex >= ts() -> tickCountPolicy() ) && startIndexArg != Py_None )
            CSP_THROW( RangeError, "buffer index out of range.  requesting data at index " << startIndex << " with buffer policy set to " << ts() -> tickCountPolicy() << " ticks in node '" << m_node -> name() << "'" );
        if( unlikely( endIndex >= ts() -> tickCountPolicy() ) )
            CSP_THROW( RangeError, "buffer index out of range.  requesting data at index " << endIndex << " with buffer policy set to " << ts() -> tickCountPolicy() << " ticks in node '" << m_node -> name() << "'" );

        startIndex = std::min( static_cast<uint32_t>( startIndex ), num_ticks() - 1 );
        endIndex   = std::min( static_cast<uint32_t>( endIndex ),   num_ticks() - 1 );
    }
    else
    {
        CSP_TRUE_OR_THROW_RUNTIME( startIndexArg == Py_None || endIndexArg == Py_None ||
                                   Py_TYPE( startIndexArg ) == Py_TYPE( endIndexArg ),
                                   "Start and end index must both be datetime or both be timedelta" );

        TimeDelta startTd, endTd;
        DateTime  startDt, endDt;

        if( startIndexArg != Py_None )
        {
            auto startDtd = fromPython<DateTimeOrTimeDelta>( startIndexArg );
            if( startDtd.isTimeDelta() )
            {
                startTd = startDtd.timedelta();
                CSP_TRUE_OR_THROW_RUNTIME( startTd.sign() <= 0, "Positive timedelta is unsupported" );

                if( unlikely( -startTd > ts() -> tickTimeWindowPolicy() ) )
                    CSP_THROW( RangeError, "buffer timedelta out of range.  requesting data at timedelta " <<
                                           PyObjectPtr::incref( startIndexArg ) << " with buffer policy set to " <<
                                           PyObjectPtr::own( toPython( ts() -> tickTimeWindowPolicy() ) ) <<
                                           " in node '" << m_node -> name() << "'" );

                startIndex = computeStartIndex( m_node -> now() + startTd, startPolicy );
            }
            else
            {
                startDt = startDtd.datetime();
                CSP_TRUE_OR_THROW_RUNTIME( startDt <= m_node -> now(), "requesting data from future time" );
                if( unlikely( m_node -> now() - startDt > ts() -> tickTimeWindowPolicy() ) )
                    CSP_THROW( RangeError, "requested buffer time out of range.  requesting datetime " << PyObjectPtr::incref( startIndexArg ) << " at time " <<
                                           PyObjectPtr::own( toPython( m_node -> now() ) ) << " with buffer time window policy set to " <<
                                           PyObjectPtr::own( toPython( ts() -> tickTimeWindowPolicy() ) )<< " in node '" << m_node -> name() << "'" );

                startIndex = computeStartIndex( startDt, startPolicy );
            }
        }

        if( endIndexArg != Py_None )
        {
            auto endDtd = fromPython<DateTimeOrTimeDelta>( endIndexArg );
            if( endDtd.isTimeDelta() )
            {
                endTd = endDtd.timedelta();
                CSP_TRUE_OR_THROW_RUNTIME( endTd.sign() <= 0, "Positive timedelta is unsupported" );

                if( startIndexArg != Py_None )
                    CSP_TRUE_OR_THROW_RUNTIME( endTd >= startTd,
                                               "Start timedelta (got " << startTd << ") must come before end timedelta (got " << endTd << ")" );

                if( unlikely( -endTd > ts() -> tickTimeWindowPolicy() ) )
                    CSP_THROW( RangeError, "buffer timedelta out of range.  requesting data at timedelta " <<
                                           PyObjectPtr::incref( endIndexArg ) << " with buffer policy set to " <<
                                           PyObjectPtr::own( toPython( ts() -> tickTimeWindowPolicy() ) ) <<
                                           " in node '" << m_node -> name() << "'" );

                endIndex = computeEndIndex( m_node -> now() + endTd, endPolicy );
            }
            else
            {
                endDt = endDtd.datetime();
                CSP_TRUE_OR_THROW_RUNTIME( endDt <= m_node -> now(), "requesting data from future time" );

                if( startIndexArg != Py_None )
                    CSP_TRUE_OR_THROW_RUNTIME( endDt >= startDt,
                                               "Start datetime (got " << startDt << ") must come before end datetime (got " << endDt << ")" );

                if( unlikely( m_node -> now() - endDt >= ts() -> tickTimeWindowPolicy() ) )
                    CSP_THROW( RangeError, "requested buffer time out of range.  requesting datetime " << PyObjectPtr::incref( endIndexArg ) << " at time " <<
                                           PyObjectPtr::own( toPython( m_node -> now() ) ) << " with buffer time window policy set to " <<
                                           PyObjectPtr::own( toPython( ts() -> tickTimeWindowPolicy() ) )<< " in node '" << m_node -> name() << "'" );

                endIndex = computeEndIndex( endDt, endPolicy );
            }

            if( endIndex == -1 )
                endIndex = startIndex + 1;
        }

        if( startPolicy == autogen::TimeIndexPolicy::EXTRAPOLATE )
        {
            if( !startTd.isNone() )
                return valuesAtIndexToNumpy( valueType, ts(), startIndex, endIndex, startPolicy, endPolicy, m_node -> now() + startTd, m_node -> now() + endTd );
            if( !startDt.isNone() )
                return valuesAtIndexToNumpy( valueType, ts(), startIndex, endIndex, startPolicy, endPolicy, startDt, endDt );
        }
    }

    return valuesAtIndexToNumpy( valueType, ts(), startIndex, endIndex, startPolicy, endPolicy );
}

static PyObject * PyInputProxy_ticked( PyInputProxy * proxy )
{
    CSP_BEGIN_METHOD;

    return toPython( proxy -> ticked() );
    CSP_RETURN_NONE;
}

static PyObject * PyInputProxy_valid( PyInputProxy * proxy )
{
    CSP_BEGIN_METHOD;

    return toPython( proxy -> valid() );
    CSP_RETURN_NONE;
}

static Py_ssize_t PyInputProxy_count( PyInputProxy * proxy )
{
    CSP_BEGIN_METHOD;

    return proxy -> count();
    CSP_RETURN_INT;
}

static void scheduler_handle_destructor( PyObject * o )
{
    Scheduler::Handle * raw = ( Scheduler::Handle * ) PyCapsule_GetPointer( o, "handle" );
    delete raw;
}

static PyObject * PyInputProxy_schedule_alarm( PyInputProxy * proxy, PyObject * args )
{
    CSP_BEGIN_METHOD;

    PyObject * timeOrDelta;
    PyObject * value;
    if( !PyArg_ParseTuple( args, "OO", &timeOrDelta, &value ) )
        return nullptr;

    //todo fixme to handle time
    auto dtd = fromPython<DateTimeOrTimeDelta>( timeOrDelta );

    Scheduler::Handle * ret = new Scheduler::Handle( proxy -> scheduleAlarm( dtd, value ) );
    return PyCapsule_New( ret, "handle", scheduler_handle_destructor );

    CSP_RETURN_NONE;
}

static PyObject * PyInputProxy_reschedule_alarm( PyInputProxy * proxy, PyObject * args )
{
    CSP_BEGIN_METHOD;

    PyObject * pyHandle;
    PyObject * timeOrDelta;
    if( !PyArg_ParseTuple( args, "OO", &pyHandle, &timeOrDelta ) )
        return nullptr;

    void * raw = PyCapsule_GetPointer( pyHandle, "handle" );
    if( !raw )
        CSP_THROW( PythonPassthrough, "" );
    auto * handle = ( Scheduler::Handle * ) raw;
    auto dtd = fromPython<DateTimeOrTimeDelta>( timeOrDelta );

    Scheduler::Handle * ret = new Scheduler::Handle( proxy -> rescheduleAlarm( *handle, dtd ) );
    return PyCapsule_New( ret, "handle", scheduler_handle_destructor );
    CSP_RETURN_NONE;
}

static PyObject * PyInputProxy_cancel_alarm( PyInputProxy * proxy, PyObject * args )
{
    CSP_BEGIN_METHOD;

    PyObject * pyHandle;
    if( !PyArg_ParseTuple( args, "O", &pyHandle ) )
        return nullptr;

    void * raw = PyCapsule_GetPointer( pyHandle, "handle" );
    if( !raw )
        CSP_THROW( PythonPassthrough, "" );
    auto * handle = ( Scheduler::Handle * ) raw;

    proxy -> cancelAlarm( *handle );
    CSP_RETURN_NONE;
}

static PyObject * PyInputProxy_make_active( PyInputProxy * proxy, PyObject * args )
{
    CSP_BEGIN_METHOD;
    return toPython( proxy -> makeActive() );
    CSP_RETURN_NONE;
}

static PyObject * PyInputProxy_make_passive( PyInputProxy * proxy, PyObject * args )
{
    CSP_BEGIN_METHOD;
    return toPython( proxy -> makePassive() );
    CSP_RETURN_NONE;
}

static PyObject * PyInputProxy_set_buffering_policy( PyInputProxy * proxy, PyObject * args, PyObject * kwargs )
{
    CSP_BEGIN_METHOD;
    PyObject * tickCount   = nullptr;
    PyObject * tickHistory = nullptr;

    static const char * kwlist[] = { "tick_count", "tick_history", nullptr };
    if( !PyArg_ParseTupleAndKeywords( args, kwargs, "|O!O", ( char ** ) kwlist,
                                      &PyLong_Type, &tickCount,
                                      &tickHistory ) )
        CSP_THROW( PythonPassthrough, "" );

    if( !tickCount && !tickHistory )
        CSP_THROW( TypeError, "csp.set_buffering_policy expected at least one of tick_count or tick_history" );

    proxy -> setBufferingPolicy( tickCount ? fromPython<int32_t>( tickCount ) : -1,
                                 tickHistory ? fromPython<TimeDelta>( tickHistory ) : TimeDelta::NONE() );
    CSP_RETURN_NONE;
}

static inline PyObject * PyInputProxy_value_at_impl( ValueType valueType, PyInputProxy * proxy, PyObject * args )
{
    CSP_BEGIN_METHOD;
    PyObject * indexArg;
    PyObject * duplicatePolicyArg;
    PyObject * defaultValueArg;
    if( !PyArg_ParseTuple( args, "OOO", &indexArg, &duplicatePolicyArg, &defaultValueArg ) )
        CSP_THROW(RuntimeException, "Invalid arguments parsed to value_at");

    return proxy->valueAt( valueType, indexArg, duplicatePolicyArg, defaultValueArg);
    CSP_RETURN_NONE;
}

static inline PyObject * PyInputProxy_values_at_impl( ValueType valueType, PyInputProxy * proxy, PyObject * args )
{
    CSP_BEGIN_METHOD;
    PyObject * startIndexArg;
    PyObject * endIndexArg;
    PyObject * startExclusiveArg;
    PyObject * endExclusiveArg;
    if( !PyArg_ParseTuple( args, "OOO!O!", &startIndexArg, &endIndexArg,
                           &PyCspEnum::PyType, &startExclusiveArg,
                           &PyCspEnum::PyType, &endExclusiveArg ) )
        CSP_THROW( RuntimeException, "Invalid arguments passed to values_at" );

    return proxy -> valuesAt( valueType, startIndexArg, endIndexArg, startExclusiveArg, endExclusiveArg );
    CSP_RETURN_NONE;
}


static PyObject * PyInputProxy_value_at( PyInputProxy * proxy, PyObject * args )
{
    return PyInputProxy_value_at_impl(ValueType::VALUE, proxy, args);
}

static PyObject * PyInputProxy_time_at(PyInputProxy *proxy, PyObject *args)
{
    return PyInputProxy_value_at_impl(ValueType::TIMESTAMP, proxy, args);
}

static PyObject * PyInputProxy_item_at(PyInputProxy *proxy, PyObject *args)
{
    return PyInputProxy_value_at_impl(ValueType::TIMESTAMP_VALUE_TUPLE, proxy, args);
}

static PyObject * PyInputProxy_values_at( PyInputProxy * proxy, PyObject * args )
{
    return PyInputProxy_values_at_impl(ValueType::VALUE, proxy, args);
}

static PyObject * PyInputProxy_times_at(PyInputProxy *proxy, PyObject *args)
{
    return PyInputProxy_values_at_impl(ValueType::TIMESTAMP, proxy, args);
}

static PyObject * PyInputProxy_items_at(PyInputProxy *proxy, PyObject *args)
{
    return PyInputProxy_values_at_impl(ValueType::TIMESTAMP_VALUE_TUPLE, proxy, args);
}

static PyMethodDef PyInputProxy_methods[] = {
    {"schedule_alarm",       (PyCFunction) PyInputProxy_schedule_alarm,       METH_VARARGS, "schedule alarm"},
    {"reschedule_alarm",     (PyCFunction) PyInputProxy_reschedule_alarm,     METH_VARARGS, "reschedule alarm"},
    {"cancel_alarm",         (PyCFunction) PyInputProxy_cancel_alarm,         METH_VARARGS, "cancel alarm"},
    {"make_active",          (PyCFunction) PyInputProxy_make_active,          METH_NOARGS,  "make input active"},
    {"make_passive",         (PyCFunction) PyInputProxy_make_passive,         METH_NOARGS,  "make input passive"},
    {"set_buffering_policy", (PyCFunction) PyInputProxy_set_buffering_policy, METH_VARARGS |
     METH_KEYWORDS, "set buffering policy"},
    {"value_at",             (PyCFunction) PyInputProxy_value_at,             METH_VARARGS, "Get historical value from time series"},
    {"time_at",              (PyCFunction) PyInputProxy_time_at,              METH_VARARGS, "Get historical timestamp from time series"},
    {"item_at",              (PyCFunction) PyInputProxy_item_at,              METH_VARARGS, "Get historical timestamp and value tuple from time series"},
    {"values_at",            (PyCFunction) PyInputProxy_values_at,            METH_VARARGS, "Get historical values from time series"},
    {"times_at",             (PyCFunction) PyInputProxy_times_at,             METH_VARARGS, "Get historical timestamps from time series"},
    {"items_at",             (PyCFunction) PyInputProxy_items_at,             METH_VARARGS, "Get historical timestamp and value tuples from time series"},
    {NULL}
};

static PySequenceMethods PyInputProxy_SeqMethods = {
    (lenfunc) PyInputProxy_count,
};

static PyNumberMethods PyInputProxy_NumberMethods = {
    0, /* binaryfunc nb_add */
    0, /* binaryfunc nb_subtract */
    0, /* binaryfunc nb_multiply */
    0, /* binaryfunc nb_remainder */
    0, /* binaryfunc nb_divmod */
    0, /* ternaryfunc nb_power */
    ( unaryfunc ) PyInputProxy_valid,  /* unaryfunc nb_negative */
    ( unaryfunc ) PyInputProxy_ticked, /* unaryfunc nb_positive */
    0, /* unaryfunc nb_absolute */
    0, /* inquiry nb_nonzero */
    0, /* unaryfunc nb_invert */
    0, /* binaryfunc nb_lshift */
    0, /* binaryfunc nb_rshift */
    0, /* binaryfunc nb_and */
    0, /* binaryfunc nb_xor */
    0, /* binaryfunc nb_or */
    0, /* unaryfunc nb_int */
    0, /* void * reserved */
    0, /* unaryfunc nb_float */
};


PyTypeObject PyInputProxy::PyType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_cspimpl.PyInputProxy",   /* tp_name */
    sizeof(PyInputProxy),      /* tp_basicsize */
    0,                         /* tp_itemsize */
    0,                         /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    0,                         /* tp_repr */
    &PyInputProxy_NumberMethods,/* tp_as_number */
    &PyInputProxy_SeqMethods,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash  */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,        /* tp_flags */
    "csp input proxy",         /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    PyInputProxy_methods,      /* tp_methods */
    0,                         /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    0,                         /* tp_init */
    0,                         /* tp_alloc */
    PyType_GenericNew,         /* tp_new */
    0,                         /* tp_free */ /* Low-level free-memory routine */
    0,                         /* tp_is_gc */ /* For PyObject_IS_GC */
    0,                         /* tp_bases */
    0,                         /* tp_mro */ /* method resolution order */
    0,                         /* tp_cache */
    0,                         /* tp_subclasses */
    0,                         /* tp_weaklist */
    0,                         /* tp_del */
    0,                         /* tp_version_tag */
    0                          /* tp_finalize */
};

REGISTER_TYPE_INIT( &PyInputProxy::PyType, "PyInputProxy" );

}
