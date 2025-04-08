#include <xmmintrin.h>
#include <pmmintrin.h>
#include <cmath>

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
		__m128 diff_simd = _mm_sub_ps(v1_simd, v2_simd);
		alignas(16) float diff[4];
		_mm_store_ps(diff, diff_simd);
		vector4 result = vector4(diff[0], diff[1], diff[2], diff[3]);
		return result;
	}

	vector4& sub(float x, float y, float z)
	{
		alignas(16) float v1[4] = { this->x(), this->y(), this->z(), this->w() };
		alignas(16) float v2[4] = { x, y, z, 0 };
		__m128 v1_simd = _mm_load_ps(v1);
		__m128 v2_simd = _mm_load_ps(v2);
		__m128 diff_simd = _mm_sub_ps(v1_simd, v2_simd);
		alignas(16) float diff[4];
		_mm_store_ps(diff, diff_simd);
		vector4 result = vector4(diff[0], diff[1], diff[2], diff[3]);
		return result;
	}

	vector4& mul(float scale)
	{
		alignas(16) float v1[4] = { this->x(), this->y(), this->z(), this->w() };
		__m128 v1_simd = _mm_load_ps(v1);
		__m128 v2_simd = _mm_set_ps(scale, scale, scale, 1);
		__m128 prod_simd = _mm_mul_ps(v1_simd, v2_simd);
		alignas(16) float prod[4];
		_mm_store_ps(prod, prod_simd);
		vector4 result = vector4(prod[0], prod[1], prod[2], prod[3]);
		return result;
	}

	vector4& mul(float scale, float w_scale)
	{
		alignas(16) float v1[4] = { this->x(), this->y(), this->z(), this->w() };
		__m128 v1_simd = _mm_load_ps(v1);
		__m128 v2_simd = _mm_set_ps(scale, scale, scale, w_scale);
		__m128 prod_simd = _mm_mul_ps(v1_simd, v2_simd);
		alignas(16) float prod[4];
		_mm_store_ps(prod, prod_simd);
		vector4 result = vector4(prod[0], prod[1], prod[2], prod[3]);
		return result;
	}

	vector4& div(float scale)
	{
		alignas(16) float v1[4] = { this->x(), this->y(), this->z(), this->w() };
		__m128 v1_simd = _mm_load_ps(v1);
		__m128 v2_simd = _mm_set_ps(scale, scale, scale, 1);
		__m128 res_simd = _mm_div_ps(v1_simd, v2_simd);
		alignas(16) float res[4];
		_mm_store_ps(res, res_simd);
		vector4 result = vector4(res[0], res[1], res[2], res[3]);
		return result;
	}

	vector4& div(float scale, float w_scale)
	{
		alignas(16) float v1[4] = { this->x(), this->y(), this->z(), this->w() };
		__m128 v1_simd = _mm_load_ps(v1);
		__m128 v2_simd = _mm_set_ps(scale, scale, scale, w_scale);
		__m128 res_simd = _mm_div_ps(v1_simd, v2_simd);
		alignas(16) float res[4];
		_mm_store_ps(res, res_simd);
		vector4 result = vector4(res[0], res[1], res[2], res[3]);
		return result;
	}

	float& dot(const vector4& other)
	{
		alignas(16) float v1[4] = { this->x(), this->y(), this->z(), this->w() };
		alignas(16) float v2[4] = { other.x(), other.y(), other.z(), other.w() };
		__m128 v1_simd = _mm_load_ps(v1);
		__m128 v2_simd = _mm_load_ps(v2);
		__m128 prod_simd = _mm_mul_ps(v1_simd, v2_simd);
		alignas(16) float prod[4];
		_mm_store_ps(prod, prod_simd);
		float dot_product = prod[0] + prod[1] + prod[2] + prod[3];
		return dot_product;
	}

	float& dot(float x, float y, float z)
	{
		alignas(16) float v1[4] = { this->x(), this->y(), this->z(), this->w() };
		alignas(16) float v2[4] = { x, y, z, 0 };
		__m128 v1_simd = _mm_load_ps(v1);
		__m128 v2_simd = _mm_load_ps(v2);
		__m128 prod_simd = _mm_mul_ps(v1_simd, v2_simd);
		alignas(16) float prod[4];
		_mm_store_ps(prod, prod_simd);
		float dot_product = prod[0] + prod[1] + prod[2] + prod[3];
		return dot_product;
	}

	float magnitude() const
	{
		alignas(16) float v[4] = { this->x(), this->y(), this->z(), this->w() };
		__m128 v_simd = _mm_load_ps(v);
		__m128 prod_simd = _mm_mul_ps(v_simd, v_simd);
		alignas(16) float prod[4];
		_mm_store_ps(prod, prod_simd);
		float dot_product = prod[0] + prod[1] + prod[2] + prod[3];
		return dot_product;
	}

	float magnitude_square() const
	{
		alignas(16) float v[4] = { this->x(), this->y(), this->z(), this->w() };
		__m128 v_simd = _mm_load_ps(v);
		__m128 prod_simd = _mm_mul_ps(v_simd, v_simd);
		alignas(16) float prod[4];
		_mm_store_ps(prod, prod_simd);
		float dot_product = prod[0] + prod[1] + prod[2] + prod[3];
		return sqrt(dot_product);
	}

	vector4& normalize()
	{
		alignas(16) float v[4] = { this->x(), this->y(), this->z(), this->w() };
		__m128 v_simd = _mm_load_ps(v);
		__m128 prod_simd = _mm_mul_ps(v_simd, v_simd);
		alignas(16) float prod[4];
		_mm_store_ps(prod, prod_simd);
		float dot_product = prod[0] + prod[1] + prod[2] + prod[3];
		float scale = sqrt(dot_product);
		__m128 v2_simd = _mm_set_ps(scale, scale, scale, 1);
		__m128 res_simd = _mm_div_ps(v_simd, v2_simd);
		alignas(16) float res[4];
		_mm_store_ps(res, res_simd);
		vector4 result = vector4(res[0], res[1], res[2], res[3]);
		return result;
	}
};