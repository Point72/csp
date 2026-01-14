/* 
 * ABI-stable C Output Adapter interface for CSP Engine
*/
#ifndef _IN_CSP_ENGINE_COUTPUTADAPTER_H
#define _IN_CSP_ENGINE_COUTPUTADAPTER_H

// C Output Adapter interface
#ifdef __cplusplus
extern "C" {
#endif

// Construction:
//   - ignore engine pointer as it is (hopefully) not needed
//   - dictionary of properties

// Execution:
//  - void executeImpl()
//  - inside executeImpl, input() -> lastValueTyped<T>() will be invoked
//        lastValueType<T>() will need to be exposed via C interface as well

// Destruction:
//  - standard destructor


typedef struct {
    /* Opaque Type internal to adapter implementation */
} OutputAdapter;


#ifdef __cplusplus
}
#endif

#endif
