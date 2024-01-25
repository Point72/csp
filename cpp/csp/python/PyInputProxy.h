#ifndef _IN_CSP_PYTHON_PYINPUTPROXY_H
#define _IN_CSP_PYTHON_PYINPUTPROXY_H

#include <csp/engine/InputId.h>
#include <csp/python/Conversions.h>
#include <csp/python/NumpyConversions.h>
#include <Python.h>

namespace csp::python
{

class PyNode;

class PyInputProxy : public PyObject
{
public:
    PyInputProxy( PyNode * node, InputId id );

    static PyInputProxy * create( PyNode * node, InputId id );

    bool ticked() const;
    bool valid() const;
    uint32_t count() const;
    uint32_t num_ticks() const;

    bool makeActive();
    bool makePassive();

    Scheduler::Handle scheduleAlarm( DateTimeOrTimeDelta timedelta, PyObject * value );
    Scheduler::Handle rescheduleAlarm( Scheduler::Handle handle, DateTimeOrTimeDelta timedelta );
    void cancelAlarm( Scheduler::Handle handle );

    void setBufferingPolicy( int32_t tickCount, TimeDelta tickHistory );

    PyObject * lastValue() const { return lastValueToPython( ts() ); }

    PyObject * valueAt(ValueType valueType, PyObject *indexArg, PyObject *duplicatePolicyArg,PyObject *defaultValueArg) const;

    //used by dynamic basket output
    void setElemId( int64_t elemId ) { m_id.elemId = elemId; }

    PyObject * valuesAt(ValueType valueType, PyObject *startIndexArg,
                        PyObject *endIndexArg,
                        PyObject *startIndexPolicyArg,
                        PyObject *endIndexPolicyArg) const;

    static PyTypeObject PyType;

private:
    int32_t computeStartIndex( DateTime startDt, autogen::TimeIndexPolicy startPolicy ) const;
    int32_t computeEndIndex( DateTime endDt, autogen::TimeIndexPolicy endPolicy ) const;

    TimeSeriesProvider * ts() const;

    PyNode * m_node;
    InputId  m_id;
};

}

#endif
