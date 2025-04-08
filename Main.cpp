#include <xmmintrin.h>

struct vector4 
{
private:
	float _x;
	float _y;
	float _z;
	float _w;

public:
	float x() const { _x; };
	float y() const { _y; };
	float z() const { _z; };
	float w() const { _w; };

	vector4(float x, float y, float z) 
	{
		_x = x;
		_y = y;
		_z = z;
		_w = 0;
	};
	vector4(float x, float y, float z, float w) 
	{
		_x = x;
		_y = y;
		_z = z;
		_w = w;
	};


	vector4& add(const vector4 &other) 
	{
		alignas(16) float v1[4] = { this->x(), this->y(), this->z(), this->w() };
		alignas(16) float v2[4] = { other.x(), other.y(), other.z(), other.w() };
		__m128 v1_simd = _mm_load_ps(v1);
		__m128 v2_simd = _mm_load_ps(v2);
		__m128 sum_simd = _mm_add_ps(v1_simd, v2_simd);
		alignas(16) float sum[4];
		_mm_store_ps(sum, sum_simd);
		vector4 result = vector4(sum[0], sum[1], sum[2], sum[3]);
		return result;
	}

	vector4& add(float x, float y, float z) 
	{
		alignas(16) float v1[4] = { this->x(), this->y(), this->z(), this->w() };
		alignas(16) float v2[4] = { x, y, z, 0 };
		__m128 v1_simd = _mm_load_ps(v1);
		__m128 v2_simd = _mm_load_ps(v2);
		__m128 sum_simd = _mm_add_ps(v1_simd, v2_simd);
		alignas(16) float sum[4];
		_mm_store_ps(sum, sum_simd);
		vector4 result = vector4(sum[0], sum[1], sum[2], sum[3]);
		return result;
	}

	vector4& sub(const vector4& other) 
	{
		alignas(16) float v1[4] = { this->x(), this->y(), this->z(), this->w() };
		alignas(16) float v2[4] = { other.x(), other.y(), other.z(), other.w() };
		__m128 v1_simd = _mm_load_ps(v1);
		__m128 v2_simd = _mm_load_ps(v2);
		__m128 sum_simd = _mm_sub_ps(v1_simd, v2_simd);
		alignas(16) float sum[4];
		_mm_store_ps(sum, sum_simd);
		vector4 result = vector4(sum[0], sum[1], sum[2], sum[3]);
		return result;
	}

	vector4& sub(float x, float y, float z)
	{
		alignas(16) float v1[4] = { this->x(), this->y(), this->z(), this->w() };
		alignas(16) float v2[4] = { x, y, z, 0 };
		__m128 v1_simd = _mm_load_ps(v1);
		__m128 v2_simd = _mm_load_ps(v2);
		__m128 sum_simd = _mm_sub_ps(v1_simd, v2_simd);
		alignas(16) float sum[4];
		_mm_store_ps(sum, sum_simd);
		vector4 result = vector4(sum[0], sum[1], sum[2], sum[3]);
		return result;
	}
};