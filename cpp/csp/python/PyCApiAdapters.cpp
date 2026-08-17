/*
 * Bridge for C API capsule-based adapters
 *
 * This module provides the glue between C API capsules (containing VTables)
 * and CSP's wiring layer (which expects InputAdapter * / OutputAdapter *).
 *
 * Usage from Python:
 *   - _c_api_push_input_adapter: Takes a capsule containing CCspPushInputAdapterVTable
 *   - _c_api_output_adapter: Takes a capsule containing CCspOutputAdapterVTable
 *   - _c_api_adapter_manager_bridge: Takes a capsule containing CCspAdapterManagerVTable
 */

#include <csp/engine/AdapterManagerExtern.h>
#include <csp/engine/PushInputAdapterExtern.h>
#include <csp/engine/OutputAdapterExtern.h>
#include <csp/engine/c/InputAdapter.h>
#include <csp/engine/c/OutputAdapter.h>
#include <csp/engine/c/AdapterManager.h>
#include <csp/python/c/PyAdapterManager.h>
#include <csp/python/c/PyInputAdapter.h>
#include <csp/python/c/PyOutputAdapter.h>
#include <csp/python/Conversions.h>
#include <csp/python/Exception.h>
#include <csp/python/InitHelper.h>
#include <csp/python/PyEngine.h>
#include <csp/python/PyInputAdapterWrapper.h>
#include <csp/python/PyOutputAdapterWrapper.h>

namespace csp::python
{

/*
 * The PushGroup is owned by a Python capsule which can be collected once wiring completes, while
 * the engine keeps the raw pointer for the whole run, so hold a reference alongside it.
 */
class CApiPushInputAdapter final : public PushInputAdapterExtern
{
public:
    CApiPushInputAdapter( Engine * engine, CspTypePtr & type, PushMode pushMode, PushGroup * group,
                          const CCspPushInputAdapterVTable & vtable, PyObjectPtr pyPushGroup )
        : PushInputAdapterExtern( engine, type, pushMode, group, vtable )
        , m_pyPushGroup( pyPushGroup )
    {
    }

private:
    PyObjectPtr m_pyPushGroup;
};

/*
 * Create a PushInputAdapterExtern from a C API capsule.
 *
 * Expected args tuple: (capsule, push_group_or_none, additional_args...)
 * The capsule contains a CCspPushInputAdapterVTable.
 */
static InputAdapter * c_api_push_input_adapter_creator( csp::AdapterManager * manager, PyEngine * pyengine, PyObject * pyType,
                                                        PushMode pushMode, PyObject * args )
{
    PyObject * capsule = nullptr;
    PyObject * pyPushGroup = nullptr;

    /* Parse args - expect (capsule, push_group_or_none) */
    if( !PyArg_ParseTuple( args, "OO", &capsule, &pyPushGroup ) )
        CSP_THROW( PythonPassthrough, "" );

    /* Validate and extract the VTable from the capsule */
    if( !PyCapsule_IsValid( capsule, CSP_C_INPUT_ADAPTER_CAPSULE_NAME ) )
        CSP_THROW( TypeError, "Expected input adapter capsule (csp.c.InputAdapterCapsule)" );

    CCspPushInputAdapterVTable * vtable = static_cast<CCspPushInputAdapterVTable *>( PyCapsule_GetPointer( capsule, CSP_C_INPUT_ADAPTER_CAPSULE_NAME ) );

    if( !vtable )
        CSP_THROW( ValueError, "Failed to extract VTable from capsule" );

    if( PyCapsule_GetContext( capsule ) == CCSP_PY_CAPSULE_TRANSFERRED )
        CSP_THROW( ValueError, "Input adapter capsule has already been used; create a new one per adapter" );

    /* Get push group if provided */
    csp::PushGroup * pushGroup = nullptr;
    if( pyPushGroup != Py_None )
    {
        pushGroup = static_cast<csp::PushGroup *>( PyCapsule_GetPointer( pyPushGroup, nullptr ) );
        if( !pushGroup )
        {
            PyErr_Clear();
            CSP_THROW( TypeError, "Expected PushGroup instance for push group" );
        }
    }

    /* Get the CSP type from Python type */
    auto & cspType = pyTypeAsCspType( pyType );

    /* Claim user_data before constructing. createOwnedObject registers the object after building
     * it, and if that registration throws, the object's destructor has already run destroy().
     * Claiming first can leak user_data if construction itself fails, but never double-frees it. */
    if( PyCapsule_SetContext( capsule, CCSP_PY_CAPSULE_TRANSFERRED ) != 0 )
        CSP_THROW( PythonPassthrough, "" );

    /* The adapter holds its own copy of the VTable. The capsule keeps ownership of its heap copy,
     * so it is still released by whoever allocated it. */
    return pyengine->engine()->createOwnedObject<CApiPushInputAdapter>( cspType, pushMode, pushGroup, *vtable,
                                                                       PyObjectPtr::incref( pyPushGroup ) );
}

/*
 * Create an OutputAdapterExtern from a C API capsule.
 *
 * Expected args tuple: (capsule, additional_args...)
 * The capsule contains a CCspOutputAdapterVTable.
 */
static OutputAdapter * c_api_output_adapter_creator(
    csp::AdapterManager * manager,
    PyEngine * pyengine,
    PyObject * args )
{
    PyObject * capsule = nullptr;
    PyObject * pyInputType = nullptr;

    /* Parse args - expect (input_type, capsule) */
    if( !PyArg_ParseTuple( args, "OO", &pyInputType, &capsule ) )
        CSP_THROW( PythonPassthrough, "" );

    /* Validate and extract the VTable from the capsule */
    if( !PyCapsule_IsValid( capsule, CSP_C_OUTPUT_ADAPTER_CAPSULE_NAME ) )
        CSP_THROW( TypeError, "Expected output adapter capsule (csp.c.OutputAdapterCapsule)" );

    CCspOutputAdapterVTable * vtable = static_cast<CCspOutputAdapterVTable *>( PyCapsule_GetPointer( capsule, CSP_C_OUTPUT_ADAPTER_CAPSULE_NAME ) );

    if( !vtable )
        CSP_THROW( ValueError, "Failed to extract VTable from capsule" );

    if( PyCapsule_GetContext( capsule ) == CCSP_PY_CAPSULE_TRANSFERRED )
        CSP_THROW( ValueError, "Output adapter capsule has already been used; create a new one per adapter" );

    /* Get the CSP type from Python type */
    auto & cspType = pyTypeAsCspType( pyInputType );

    /* See the input adapter creator for why the capsule is marked before construction */
    if( PyCapsule_SetContext( capsule, CCSP_PY_CAPSULE_TRANSFERRED ) != 0 )
        CSP_THROW( PythonPassthrough, "" );

    return pyengine->engine()->createOwnedObject<OutputAdapterExtern>( cspType, *vtable );
}

REGISTER_INPUT_ADAPTER( _c_api_push_input_adapter, c_api_push_input_adapter_creator );
REGISTER_OUTPUT_ADAPTER( _c_api_output_adapter, c_api_output_adapter_creator );

/*
 * Bridge a C API adapter manager capsule to CSP's internal format.
 *
 * Python signature: _c_api_adapter_manager_bridge(engine, c_api_capsule) -> csp_capsule
 *
 * This takes a capsule containing CCspAdapterManagerVTable (from C API) and
 * returns a capsule containing AdapterManagerExtern* that CSP's wiring layer expects.
 */
static PyObject * c_api_adapter_manager_bridge( PyObject * /* self */, PyObject * args )
{
    CSP_BEGIN_METHOD;

    PyEngine * pyEngine = nullptr;
    PyObject * cApiCapsule = nullptr;

    if( !PyArg_ParseTuple( args, "O!O",
                           &PyEngine::PyType, &pyEngine,
                           &cApiCapsule ) )
        CSP_THROW( PythonPassthrough, "" );

    /* Validate the C API capsule */
    if( !PyCapsule_IsValid( cApiCapsule, CSP_C_ADAPTER_MANAGER_CAPSULE_NAME ) )
        CSP_THROW( TypeError, "Expected C API adapter manager capsule (csp.c.AdapterManagerCapsule)" );

    CCspAdapterManagerVTable * vtable = static_cast<CCspAdapterManagerVTable *>( PyCapsule_GetPointer( cApiCapsule, CSP_C_ADAPTER_MANAGER_CAPSULE_NAME ) );

    if( !vtable )
        CSP_THROW( ValueError, "Failed to extract VTable from C API capsule" );

    if( PyCapsule_GetContext( cApiCapsule ) == CCSP_PY_CAPSULE_TRANSFERRED )
        CSP_THROW( ValueError, "Adapter manager capsule has already been used; create a new one per manager" );

    /* See the input adapter creator for why the capsule is marked before construction */
    if( PyCapsule_SetContext( cApiCapsule, CCSP_PY_CAPSULE_TRANSFERRED ) != 0 )
        CSP_THROW( PythonPassthrough, "" );

    /* Create the AdapterManagerExtern owned by the engine */
    auto * adapterMgr = pyEngine->engine()->createOwnedObject<AdapterManagerExtern>( *vtable );

    /* Return a capsule with the name CSP's wiring layer expects */
    return PyCapsule_New( adapterMgr, "adapterMgr", nullptr );

    CSP_RETURN_NULL;
}

REGISTER_MODULE_METHOD( "_c_api_adapter_manager_bridge", c_api_adapter_manager_bridge, METH_VARARGS,
                        "Bridge a C API adapter manager capsule to CSP format" );

}
