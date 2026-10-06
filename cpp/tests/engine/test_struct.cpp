#include <csp/engine/CspType.h>
#include <csp/engine/Struct.h>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>

using namespace csp;

namespace
{

std::shared_ptr<StructMeta> makeLeafMeta()
{
    StructMeta::Fields fields = { std::make_shared<StringStructField>( CspType::STRING(), "name", false ),
                                  std::make_shared<Int64StructField>( "value", false ) };
    return std::make_shared<StructMeta>( "Leaf", fields, false );
}

StructPtr makeLeaf( const std::shared_ptr<StructMeta> & meta )
{
    StructPtr s = meta -> create();
    meta -> getMetaField<std::string>( "name", "test" ) -> setValue( s.get(), std::string( "long enough to not fit in sso buffer" ) );
    meta -> getMetaField<int64_t>( "value", "test" ) -> setValue( s.get(), 123 );
    return s;
}

}

TEST( StructTest, test_struct_lifetime_releases_meta )
{
    auto meta = makeLeafMeta();
    auto baseline = meta.use_count();

    for( int i = 0; i < 1000; ++i )
    {
        StructPtr s = makeLeaf( meta );
        StructPtr copy = s -> copy();
        StructPtr deepcopy = s -> deepcopy();
    }

    ASSERT_EQ( meta.use_count(), baseline );
}

TEST( StructTest, test_live_structs_hold_meta )
{
    auto meta = makeLeafMeta();
    auto baseline = meta.use_count();

    std::vector<StructPtr> structs;
    for( int i = 0; i < 10; ++i )
        structs.push_back( makeLeaf( meta ) );
    ASSERT_EQ( meta.use_count(), baseline + 10 );

    structs.clear();
    ASSERT_EQ( meta.use_count(), baseline );
}

TEST( StructTest, test_nested_struct_lifetime_releases_metas )
{
    auto leafMeta = makeLeafMeta();
    StructMeta::Fields fields = { std::make_shared<StructStructField>( std::make_shared<CspStructType>( leafMeta ), "leaf", false ) };
    auto outerMeta = std::make_shared<StructMeta>( "Outer", fields, false );
    auto leafField = outerMeta -> getMetaField<StructPtr>( "leaf", "test" );
    auto leafBaseline = leafMeta.use_count();
    auto outerBaseline = outerMeta.use_count();

    for( int i = 0; i < 1000; ++i )
    {
        StructPtr s = outerMeta -> create();
        leafField -> setValue( s.get(), makeLeaf( leafMeta ) );
        StructPtr copy = s -> copy();
        StructPtr deepcopy = s -> deepcopy();
    }

    ASSERT_EQ( leafMeta.use_count(), leafBaseline );
    ASSERT_EQ( outerMeta.use_count(), outerBaseline );
}

TEST( StructTest, test_derived_struct_lifetime_releases_metas )
{
    auto baseMeta = makeLeafMeta();
    StructMeta::Fields fields = { std::make_shared<StringStructField>( CspType::STRING(), "extra", false ) };
    auto derivedMeta = std::make_shared<StructMeta>( "Derived", fields, false, baseMeta );
    auto baseBaseline = baseMeta.use_count();
    auto derivedBaseline = derivedMeta.use_count();

    for( int i = 0; i < 1000; ++i )
    {
        StructPtr s = makeLeaf( derivedMeta );
        derivedMeta -> getMetaField<std::string>( "extra", "test" ) -> setValue( s.get(), std::string( "another string long enough to allocate" ) );
        StructPtr copy = s -> copy();
        StructPtr deepcopy = s -> deepcopy();
    }

    ASSERT_EQ( baseMeta.use_count(), baseBaseline );
    ASSERT_EQ( derivedMeta.use_count(), derivedBaseline );
}
