#include <gtest/gtest.h>
#include <csp/engine/Dictionary.h>

using namespace csp;

namespace csp
{
DialectGenericType::DialectGenericType() {}
DialectGenericType::~DialectGenericType() {}
DialectGenericType::DialectGenericType( const DialectGenericType &rhs ) {}
DialectGenericType::DialectGenericType( DialectGenericType &&rhs ) {}
DialectGenericType &DialectGenericType::operator=( const DialectGenericType &rhs ) { return *this; }
DialectGenericType &DialectGenericType::operator=( DialectGenericType &&rhs ) { return *this; }
bool DialectGenericType::operator==( const DialectGenericType &rhs ) const { return true; }
size_t DialectGenericType::hash() const { return 0; }
}

TEST( Dictionary, test_basic_functionality )
{
    Dictionary d1;
    ASSERT_TRUE( d1.insert( "123", 123 ) );
    ASSERT_FALSE( d1.insert( "123", 456 ) );
    ASSERT_TRUE( d1.insert( "bool", true ) );
    ASSERT_TRUE( d1.insert( "int32", (int32_t ) 456 ) );
    ASSERT_TRUE( d1.insert( "uint32", (uint32_t ) 789 ) );
    ASSERT_TRUE( d1.insert( "int64", int64_t(1) << 45 ) );
    ASSERT_TRUE( d1.insert( "uint64", ( uint64_t(1) << 63 ) + 1  ) );
    ASSERT_TRUE( d1.insert( "double", 123.456 ) );
    ASSERT_TRUE( d1.insert( "string", std::string( "HOWDY!" ) ) );
    ASSERT_TRUE( d1.insert( "string2", "HOWDY2!" ) );
    ASSERT_TRUE( d1.insert( "time", DateTime( 2020, 4, 22, 10, 26 ) ) );
    ASSERT_TRUE( d1.insert( "timedelta", TimeDelta::fromMilliseconds( 123 ) ) );

    ASSERT_TRUE( d1.exists( "123" ) );
    ASSERT_FALSE( d1.exists( "1234" ) );

    ASSERT_EQ( d1.get<int32_t>( "123" ), 123 );
    ASSERT_EQ( d1.get<bool>(    "bool" ), true );
    ASSERT_EQ( d1.get<int32_t>( "int32" ), 456 );
    ASSERT_EQ( d1.get<int64_t>( "int64" ), int64_t(1) << 45 );
    ASSERT_EQ( d1.get<uint32_t>( "uint32" ), 789 );
    ASSERT_EQ( d1.get<uint64_t>( "uint64" ), ( uint64_t(1) << 63 ) + 1 );
    ASSERT_EQ( d1.get<double>(  "double" ), 123.456 );
    ASSERT_EQ( d1.get<DateTime>(  "time" ), DateTime( 2020, 4, 22, 10, 26 ) );
    ASSERT_EQ( d1.get<TimeDelta>(  "timedelta" ), TimeDelta::fromMilliseconds( 123 ) );
    ASSERT_EQ( d1.get<std::string>( "string" ), "HOWDY!" );
    ASSERT_EQ( d1.get<std::string>( "string2" ), "HOWDY2!" );

    ASSERT_TRUE( d1.update( "double", "double_string!" ) );
    ASSERT_EQ( d1.get<std::string>( "double" ), "double_string!" );
}

TEST( Dictionary, test_comp_hash )
{
    Dictionary d1;
    Dictionary d2;

    std::vector<Dictionary::Data> vec;
    vec.emplace_back( "a" );
    vec.emplace_back( "b" );
    vec.emplace_back( "c" );
    
    d1.insert( "123", 0 );
    d1.insert( "bool", true );
    d1.insert( "int32", (int32_t ) 456 );
    d1.insert( "int64", int64_t(1) << 45 );
    d1.insert( "double", 123.456 );
    d1.insert( "string", std::string( "HOWDY!" ) );
    d1.insert( "string2", "HOWDY2!" );
    d1.insert( "time", DateTime( 2020, 4, 22, 10, 26 ) );
    d1.insert( "timedelta", TimeDelta::fromMilliseconds( 123 ) );
    d1.insert( "vector<string>", vec );
    // When we update should be the same as if inserted at first as 123
    d1.update( "123", 123 );
    ASSERT_EQ(d1.begin().key(), "123");


    //do d2 backwards to ensure ordering in comp / hash is good
    d2.insert( "vector<string>", vec );
    d2.insert( "timedelta", TimeDelta::fromMilliseconds( 123 ) );
    d2.insert( "time", DateTime( 2020, 4, 22, 10, 26 ) );
    d2.insert( "string2", "HOWDY2!" );
    d2.insert( "string", std::string( "HOWDY!" ) );
    d2.insert( "double", 123.456 );
    d2.insert( "int64", int64_t(1) << 45 );
    d2.insert( "int32", (int32_t ) 456 );
    d2.insert( "bool", true );
    d2.insert( "123", 123 );

    ASSERT_EQ( d1, d2 );
    ASSERT_EQ( d2, d1 );

    ASSERT_NE( d1.hash(), 0 );
    ASSERT_EQ( d1.hash(), d2.hash() );

    d2.update( "123", 1234 );
    ASSERT_NE( d1.hash(), d2.hash() );

    d2.update( "123", 123 );
    ASSERT_EQ( d1.hash(), d2.hash() );

    d2.insert( "456", 456 );
    ASSERT_NE( d1.hash(), d2.hash() );
}

TEST( Dictionary, test_type_coercion )
{
    Dictionary d1;

    std::vector<Dictionary::Data> vec;
    vec.emplace_back( "a" );
    vec.emplace_back( "b" );
    vec.emplace_back( "c" );

    d1.insert( "d",   123.456 );
    d1.insert( "i64", (int64_t) 123 );
    d1.insert( "i32", (int32_t) 456 );
    d1.insert( "u32", (uint32_t) 789 );
    d1.insert( "u64", (uint64_t) 111 );
    d1.insert( "vec", vec );

    ASSERT_EQ( d1.get<double>( "d" ), 123.456 );
    ASSERT_EQ( d1.get<double>( "i64" ), 123.0 );
    ASSERT_EQ( d1.get<double>( "i32" ), 456.0 );
    ASSERT_EQ( d1.get<double>( "u64" ), 111.0 );
    ASSERT_EQ( d1.get<double>( "u32" ), 789.0 );

    ASSERT_EQ( d1.get<int32_t>(  "u32" ), 789 );
    ASSERT_EQ( d1.get<uint32_t>( "i32" ), 456 );

    ASSERT_EQ( d1.get<int64_t>( "i32" ), 456 );
    ASSERT_EQ( d1.get<int64_t>( "u32" ), 789 );
    ASSERT_EQ( d1.get<int64_t>( "u64" ), 111 );

    ASSERT_EQ( d1.get<uint64_t>( "i64" ), 123 );
    ASSERT_EQ( d1.get<uint64_t>( "i32" ), 456 );
    ASSERT_EQ( d1.get<uint64_t>( "u32" ), 789 );

    ASSERT_EQ( d1.get<std::vector<Dictionary::Data>>( "vec" ), vec );
}

TEST( Dictionary, test_iteration )
{
    Dictionary d1;
    d1.insert( "d",   123.456 );
    d1.insert( "i64", (int64_t) 123 );
    d1.insert( "i32", (int32_t) 456 );
    d1.insert( "string", std::string( "HOWDY!" ) );

    for( auto it = d1.begin(); it != d1.end(); ++it )
    {
        if( it.key() == "d" )
            ASSERT_EQ( it.value<double>(), 123.456 );
        else if( it.key() == "i64" )
            ASSERT_EQ( it.value<int64_t>(), 123 );
        else if( it.key() == "i32" )
            ASSERT_EQ( it.value<int32_t>(), 456 );
        else if( it.key() == "string" )
            ASSERT_EQ( it.value<std::string>(), "HOWDY!" );
        else
            ASSERT_TRUE( false );
    }
}

TEST( Dictionary, test_composition )
{
    DictionaryPtr subdict = std::make_shared<Dictionary>();
    subdict -> insert( "sub1", 123.456 );
    subdict -> insert( "sub2", "sub2" );

    Dictionary d1;
    d1.insert( "i", 123 );
    d1.insert( "sub", subdict );

    ASSERT_EQ( d1.get<int32_t>( "i" ), 123 );
    ASSERT_EQ( d1.get<DictionaryPtr>( "sub" ) -> get<double>( "sub1" ), 123.456 );
    ASSERT_EQ( d1.get<DictionaryPtr>( "sub" ) -> get<std::string>( "sub2" ), "sub2" );
}
