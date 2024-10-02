#include <csp/engine/Engine.h>
#include <csp/engine/CspType.h>
#include <csp/engine/CppNode.h>
#include <csp/python/PyCppNode.h>
#include <algorithm>
#include <string>

class piglatin : public csp::CppNode {
  [[maybe_unused]] CSP csp;
public:     
    const char * name() const override { return "piglatin"; }
    static csp::CppNode * create( csp::Engine * engine, const csp::CppNode::NodeDef & nodedef)
    {
        auto * out = engine -> createOwnedObject<piglatin>(nodedef);
        out -> resetNodeDef();
        return out;
    }

    piglatin( csp::Engine * engine, const csp::CppNode::NodeDef & nodedef ) : csp::CppNode( engine, nodedef ) {}

    typename InputTypeHelper<std::string>::type x{"x", *this};
    csp::CppNode::Scalar<bool> capitalize{"capitalize", *this};

    typename OutputTypeHelper<std::string>::type __unnamed_output{"",*this}; auto & unnamed_output() { return __unnamed_output; }

    void start() override {}

    void executeImpl() override
    {
      if(csp.ticked(x) && csp.valid(x))
      {
        std::string str = x.lastValue();
        if (capitalize)
        {
          std::transform(str.begin(), str.end(), str.begin(), ::toupper);
        }
        do {
          __unnamed_output.output(str.substr(1, std::string::npos) + str[0] + (capitalize ? "AY" : "ay")); return;
        } while(0);
      }
    }
};

csp::CppNode * piglatin_create_method(csp::Engine * engine, const csp::CppNode::NodeDef & nodedef ) { return piglatin::create(engine, nodedef); }

static PyObject * piglatin_cppnode_create( PyObject * module, PyObject * args )
{
    try {
      return csp::python::pycppnode_create( args, piglatin_create_method );
    } catch( csp::python::PythonPassthrough & err ) { err.restore(); return nullptr; } CSP_CATCH_HELPERS( return nullptr; ); return nullptr;
    
}


static PyModuleDef _piglatin_module = {
    PyModuleDef_HEAD_INIT,
    "_piglatin",
    "_piglatin c++ module",
    -1,
    NULL, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit__piglatin(void)
{
    PyObject* m;

    m = PyModule_Create(&_piglatin_module);
    if(m == NULL)
        return NULL;

    csp::python::InitHelper::instance().registerCallback(
      csp::python::InitHelper::moduleMethod("piglatin", piglatin_cppnode_create, METH_VARARGS, "piglatin")
    );
    if(!csp::python::InitHelper::instance().execute(m))
        return NULL;

    return m;
}
