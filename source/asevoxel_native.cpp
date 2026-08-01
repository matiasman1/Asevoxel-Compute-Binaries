// Unified minimal native acceleration (transform_voxel + calculate_face_visibility)

// NOMINMAX must be defined before any Windows headers to prevent min/max macro conflicts
#define NOMINMAX

extern "C" {
#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>
}

#define NATIVE_EXPORT
#include "render/shaders/native_shader_api.h"
#include "render/shaders/native_shader_loader.cpp"

#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <array>
#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
static constexpr double PI = 3.14159265358979323846;

static double getNum(lua_State* L, int idx, const char* k, double def=0.0) {
  double r = def;
  lua_getfield(L, idx, k);
  if (lua_isnumber(L, -1)) r = lua_tonumber(L, -1);
  lua_pop(L,1);
  return r;
}

static int getFieldInteger(lua_State* L, int idx, const char* k, int def=0) {
  int r = def;
  lua_getfield(L, idx, k);
  if (lua_isnumber(L, -1)) r = (int)lua_tointeger(L, -1);
  lua_pop(L,1);
  return r;
}

// transform_voxel(voxelTbl, paramsTbl)
static int l_transform_voxel(lua_State* L) {
  if (!lua_istable(L,1) || !lua_istable(L,2)) {
    lua_pushnil(L); lua_pushstring(L,"expected (voxel, params) tables"); return 2;
  }
  double vx = getNum(L,1,"x",0.0);
  double vy = getNum(L,1,"y",0.0);
  double vz = getNum(L,1,"z",0.0);

  double mx=0,my=0,mz=0;
  lua_getfield(L,2,"middlePoint");
  if (lua_istable(L,-1)) {
    mx = getNum(L,-1,"x",0.0);
    my = getNum(L,-1,"y",0.0);
    mz = getNum(L,-1,"z",0.0);
  }
  lua_pop(L,1);

  double xr = getNum(L,2,"xRotation",0.0);
  double yr = getNum(L,2,"yRotation",0.0);
  double zr = getNum(L,2,"zRotation",0.0);

  double x = vx - mx;
  double y = vy - my;
  double z = vz - mz;

  const double RX = xr * PI/180.0;
  const double RY = yr * PI/180.0;
  const double RZ = zr * PI/180.0;
  const double cx = std::cos(RX), sx = std::sin(RX);
  const double cy = std::cos(RY), sy = std::sin(RY);
  const double cz = std::cos(RZ), sz = std::sin(RZ);

  { double y2 = y*cx - z*sx; double z2 = y*sx + z*cx; y = y2; z = z2; }
  { double x2 = x*cy + z*sy; double z3 = -x*sy + z*cy; x = x2; z = z3; }
  { double x3 = x*cz - y*sz; double y3 = x*sz + y*cz; x = x3; y = y3; }

  x += mx; y += my; z += mz;

  lua_createtable(L,0,4);
  lua_pushnumber(L,x); lua_setfield(L,-2,"x");
  lua_pushnumber(L,y); lua_setfield(L,-2,"y");
  lua_pushnumber(L,z); lua_setfield(L,-2,"z");

  // Pass color table through (no deep clone needed for current usage)
  lua_getfield(L,1,"color");
  if (lua_istable(L,-1)) {
    lua_setfield(L,-2,"color");
  } else {
    lua_pop(L,1);
  }
  return 1;
}

// calculate_face_visibility(voxel, cameraPos, orthBool, rotationParams)
static int l_calculate_face_visibility(lua_State* L) {
  if (!lua_istable(L,1) || !lua_istable(L,2) || !lua_istable(L,4)) {
    lua_pushnil(L); lua_pushstring(L,"args: voxel, cameraPos, orthBool, rotationParams"); return 2;
  }
  double vx = getNum(L,1,"x",0.0);
  double vy = getNum(L,1,"y",0.0);
  double vz = getNum(L,1,"z",0.0);
  double cxPos = getNum(L,2,"x",0.0);
  double cyPos = getNum(L,2,"y",0.0);
  double czPos = getNum(L,2,"z",0.0);
  double xr = getNum(L,4,"xRotation",0.0);
  double yr = getNum(L,4,"yRotation",0.0);
  double zr = getNum(L,4,"zRotation",0.0);
  double voxelSize = getNum(L,4,"voxelSize",1.0);
  if (voxelSize < 0.001) voxelSize = 0.001;

  const double RX = xr * PI/180.0;
  const double RY = yr * PI/180.0;
  const double RZ = zr * PI/180.0;
  const double cx = std::cos(RX), sx = std::sin(RX);
  const double cy = std::cos(RY), sy = std::sin(RY);
  const double cz = std::cos(RZ), sz = std::sin(RZ);

  double vcx = std::floor(vx + 0.5);
  double vcy = std::floor(vy + 0.5);
  double vcz = std::floor(vz + 0.5);
  double vxv = cxPos - vcx;
  double vyv = cyPos - vcy;
  double vzv = czPos - vcz;
  double mag = std::sqrt(vxv*vxv + vyv*vyv + vzv*vzv);
  if (mag > 1e-4) { vxv/=mag; vyv/=mag; vzv/=mag; }

  struct Face { const char* name; double nx,ny,nz; };
  static const Face faces[] = {
    {"front",0,0,1},{"back",0,0,-1},{"right",1,0,0},
    {"left",-1,0,0},{"top",0,1,0},{"bottom",0,-1,0}
  };

  lua_createtable(L,0,6);
  // Use slightly negative threshold to ensure edge-on faces are consistently visible
  // This prevents jagged edges at certain rotation angles where faces are nearly perpendicular
  double threshold = -0.001;

  for (auto &f : faces) {
    double x1=f.nx, y1=f.ny, z1=f.nz;
    { double y2 = y1*cx - z1*sx; double z2 = y1*sx + z1*cx; y1=y2; z1=z2; }
    { double x2 = x1*cy + z1*sy; double z3 = -x1*sy + z1*cy; x1=x2; z1=z3; }
    { double x3 = x1*cz - y1*sz; double y3 = x1*sz + y1*cz; x1=x3; y1=y3; }
    double dot = x1*vxv + y1*vyv + z1*vzv;
    lua_pushboolean(L, dot > threshold);
    lua_setfield(L,-2,f.name);
  }
  return 1;
}

// --------------------------- BASIC RENDERER (Phase 2+3) -----------------------
// Lua: render_basic(voxels, params)
// See diff description for full parameter/return spec.

namespace {
struct Voxel {
  float x,y,z;
  unsigned char r,g,b,a;
};
struct FacePoly {
  float x[4];
  float y[4];
  float depth;
  unsigned char r,g,b,a;
  // Tiebreaker fields for stable sorting when depths are equal
  float voxelX, voxelY, voxelZ;
  int faceIdx;
  // Edit-mode ID map: original voxel array index (1-based to match Lua)
  uint32_t voxelId;
};

// Basic shading formula — matches basic.lua / basic.cpp / kernel.h's shade_basic exactly.
static inline float basicBrightness(float dot,float lightPct,float shadePct){
  float t = (dot + 1.0f) * 0.5f;
  float brightness = shadePct + (lightPct - shadePct) * t;
  return std::max(0.0f, std::min(1.0f, brightness / 100.0f));
}

static void rasterQuad(const FacePoly& poly,int W,int H,std::vector<unsigned char>& buf,
                       std::vector<uint32_t>* idMap=nullptr){
  float minY=poly.y[0], maxY=poly.y[0];
  for(int i=1;i<4;i++){ if(poly.y[i]<minY)minY=poly.y[i]; if(poly.y[i]>maxY)maxY=poly.y[i]; }
  int y0=(int)std::floor(minY);
  int y1=(int)std::ceil (maxY);
  if (y0<0) y0=0;
  if (y1>H) y1=H;
  struct Edge{float x0,y0,x1,y1;};
  Edge edges[4];
  for(int i=0;i<4;i++){
    int j=(i+1)&3;
    float x0=poly.x[i], y0p=poly.y[i];
    float x1=poly.x[j], y1p=poly.y[j];
    if (y0p<y1p) edges[i]={x0,y0p,x1,y1p};
    else edges[i]={x1,y1p,x0,y0p};
  }
  // Encode voxelId + faceIdx into a single uint32 for the ID map:
  // bits [0..23] = voxelId (up to 16M voxels), bits [24..26] = faceIdx (0-5)
  // voxelId==0 means preview/ghost voxel — never write to idMap so real entries aren't erased.
  bool writeId = idMap && (poly.voxelId != 0);
  uint32_t idVal = 0;
  if (writeId) {
    idVal = (poly.voxelId & 0x00FFFFFFu) | (((uint32_t)poly.faceIdx & 0x7u) << 24);
  }
  for(int y=y0;y<y1;y++){
    float scan = y+0.5f;
    std::vector<float> xs;
    for(int e=0;e<4;e++){
      auto &E=edges[e];
      if (scan>=E.y0 && scan<E.y1){
        float t=(scan-E.y0)/(E.y1-E.y0);
        xs.push_back(E.x0 + (E.x1-E.x0)*t);
      }
    }
    if (xs.size() < 2) continue;
    std::sort(xs.begin(), xs.end());
    for(size_t k=0; k + 1 < xs.size(); k += 2){
      float xa = xs[k], xb = xs[k+1];
      if (xa>xb) std::swap(xa,xb);
      int ix0=(int)std::floor(xa+0.5f);
      int ix1=(int)std::floor(xb-0.5f);
      if (ix0<0) ix0=0;
      if (ix1>=W) ix1=W-1;
      for(int x=ix0;x<=ix1;x++){
        size_t off=((size_t)y*W + x)*4;
        buf[off+0]=poly.r; buf[off+1]=poly.g; buf[off+2]=poly.b; buf[off+3]=poly.a;
        if (writeId) (*idMap)[(size_t)y*W + x] = idVal;
      }
    }
  }
}

static inline bool faceVisible(float nx,float ny,float nz,
                               float vx,float vy,float vz,
                               float threshold){
  return (nx*vx + ny*vy + nz*vz) > threshold;
}
static inline void rotateNormal(float &x,float &y,float &z,
                                float cx,float sx,float cy,float sy,float cz,float sz){
  float y2=y*cx - z*sx; float z2=y*sx + z*cx; y=y2; z=z2;
  float x2=x*cy + z*sy; float z3=-x*sy + z*cy; x=x2; z=z3;
  float x3=x*cz - y*sz; float y3=x*sz + y*cz; x=x3; y=y3;
}

static const int FACE_IDX[6][4]={
  {5,6,7,8},{2,1,4,3},{6,2,3,7},{1,5,8,4},{8,7,3,4},{1,2,6,5}
};
static const float LOCAL_FACE_NORMALS[6][3]={
  {0,0,1},{0,0,-1},{1,0,0},{-1,0,0},{0,1,0},{0,-1,0}
};
static const float UNIT_VERTS[8][3]={
  {-0.5f,-0.5f,-0.5f},{ 0.5f,-0.5f,-0.5f},{ 0.5f, 0.5f,-0.5f},{-0.5f, 0.5f,-0.5f},
  {-0.5f,-0.5f, 0.5f},{ 0.5f,-0.5f, 0.5f},{ 0.5f, 0.5f, 0.5f},{-0.5f, 0.5f, 0.5f}
};

// Convert internal face index (matches LOCAL_FACE_NORMALS/neigh[] order:
// 0=front,1=back,2=right,3=left,4=top,5=bottom) to the native shader API's
// face_dir encoding (0=+x,1=-x,2=+y,3=-y,4=+z,5=-z) expected by
// native_face_data_t.face_dir (see faceshade.cpp's face_dir_to_internal()).
static inline int internalFaceToApiDir(int f) {
  static const int kInternalToApi[6] = {4, 5, 0, 1, 2, 3};
  return (f >= 0 && f < 6) ? kInternalToApi[f] : 4;
}

struct OrthoCacheKey{int xr,yr,zr; float size; bool operator==(const OrthoCacheKey&o)const{
  return xr==o.xr && yr==o.yr && zr==o.zr && size==o.size;
}};
struct OrthoCacheKeyHash{
  std::size_t operator()(OrthoCacheKey const& k) const noexcept{
    std::size_t h=1469598103934665603ull;
    auto mix=[&](std::size_t v){ h^=v; h*=1099511628211ull; };
    mix(std::hash<int>()(k.xr));
    mix(std::hash<int>()(k.yr));
    mix(std::hash<int>()(k.zr));
    mix(std::hash<int>()((int)(k.size*1000.f)));
    return h;
  }
};
struct RotatedFaceTemplate{
  float vx[6][4];
  float vy[6][4];
  float fnx[6],fny[6],fnz[6];
};
static std::unordered_map<OrthoCacheKey,RotatedFaceTemplate,OrthoCacheKeyHash> g_orthoCache;

static const RotatedFaceTemplate& getOrthoTemplate(int xr,int yr,int zr,float size){
  OrthoCacheKey key{((xr%360)+360)%360,((yr%360)+360)%360,((zr%360)+360)%360,size};
  auto it=g_orthoCache.find(key);
  if(it!=g_orthoCache.end()) return it->second;
  double RX=key.xr*PI/180.0;
  double RY=key.yr*PI/180.0;
  double RZ=key.zr*PI/180.0;
  float cx=(float)std::cos(RX), sx=(float)std::sin(RX);
  float cy=(float)std::cos(RY), sy=(float)std::sin(RY);
  float cz=(float)std::cos(RZ), sz=(float)std::sin(RZ);
  RotatedFaceTemplate tmpl{};
  for(int f=0;f<6;f++){
    float nx=LOCAL_FACE_NORMALS[f][0];
    float ny=LOCAL_FACE_NORMALS[f][1];
    float nz=LOCAL_FACE_NORMALS[f][2];
    rotateNormal(nx,ny,nz,cx,sx,cy,sy,cz,sz);
    tmpl.fnx[f]=nx; tmpl.fny[f]=ny; tmpl.fnz[f]=nz;
  }
  float rvx[8], rvy[8]; // removed unused rvz
  for(int v=0;v<8;v++){
    float x=UNIT_VERTS[v][0]*size;
    float y=UNIT_VERTS[v][1]*size;
    float z=UNIT_VERTS[v][2]*size;
    rotateNormal(x,y,z,cx,sx,cy,sy,cz,sz);
    rvx[v]=x; rvy[v]=y; // rvz was unused, remove assignment
  }
  for(int f=0;f<6;f++){
    for(int i=0;i<4;i++){
      int idx=FACE_IDX[f][i]-1;
      tmpl.vx[f][i]=rvx[idx];
      tmpl.vy[f][i]=rvy[idx];
    }
  }
  return g_orthoCache.emplace(key,tmpl).first->second;
}

// ---------------------------------------------------------------------------
// Helper: read optional "globalBounds" table from Lua params.
// When present, overrides per-frame min/max for camera setup, and computes
// effectiveSpan (max of model AABB span, 2*max rotation radius around pivot).
// ---------------------------------------------------------------------------
struct CameraBounds {
  float camMinX, camMaxX, camMinY, camMaxY, camMinZ, camMaxZ;
  float sizeX, sizeY, sizeZ, maxDim;
  float effectiveSpan;
};

static CameraBounds computeCameraBounds(lua_State* L, int paramsIdx,
    float minX, float maxX, float minY, float maxY, float minZ, float maxZ,
    float midX, float midY, float midZ)
{
  CameraBounds cb;
  // Try globalBounds from params
  lua_getfield(L, paramsIdx, "globalBounds");
  if (lua_istable(L, -1)) {
    cb.camMinX = (float)getNum(L, -1, "minX", minX);
    cb.camMaxX = (float)getNum(L, -1, "maxX", maxX);
    cb.camMinY = (float)getNum(L, -1, "minY", minY);
    cb.camMaxY = (float)getNum(L, -1, "maxY", maxY);
    cb.camMinZ = (float)getNum(L, -1, "minZ", minZ);
    cb.camMaxZ = (float)getNum(L, -1, "maxZ", maxZ);
  } else {
    cb.camMinX = minX; cb.camMaxX = maxX;
    cb.camMinY = minY; cb.camMaxY = maxY;
    cb.camMinZ = minZ; cb.camMaxZ = maxZ;
  }
  lua_pop(L, 1);
  cb.sizeX = cb.camMaxX - cb.camMinX + 1.f;
  cb.sizeY = cb.camMaxY - cb.camMinY + 1.f;
  cb.sizeZ = cb.camMaxZ - cb.camMinZ + 1.f;
  cb.maxDim = std::max(cb.sizeX, std::max(cb.sizeY, cb.sizeZ));
  // Compute rotation radius: max distance from pivot to any cameraBounds corner.
  float maxRotRadius = 0.f;
  for (int ix = 0; ix < 2; ++ix) {
    float bx = (ix == 0) ? cb.camMinX : cb.camMaxX;
    for (int iy = 0; iy < 2; ++iy) {
      float by = (iy == 0) ? cb.camMinY : cb.camMaxY;
      for (int iz = 0; iz < 2; ++iz) {
        float bz = (iz == 0) ? cb.camMinZ : cb.camMaxZ;
        float dx = bx - midX, dy = by - midY, dz = bz - midZ;
        float r = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (r > maxRotRadius) maxRotRadius = r;
      }
    }
  }
  cb.effectiveSpan = std::max(cb.maxDim, 2.f * maxRotRadius);
  return cb;
}

// ============================================================================
// Helper: read _previewVoxels from params table
// Returns vector of Voxel structs + ghost alpha (50%).
// Preview voxels come from edit tools (shape, pencil batch, eraser, cursor).
// ============================================================================
static std::vector<Voxel> readPreviewVoxels(lua_State* L, int paramsIdx) {
  std::vector<Voxel> pvs;
  lua_getfield(L, paramsIdx, "_previewVoxels");
  if (!lua_istable(L, -1)) {
    lua_pop(L, 1);
    return pvs;
  }
  size_t pcount = lua_rawlen(L, -1);
  pvs.reserve(pcount);
  int tbl = lua_gettop(L);
  for (size_t i = 1; i <= pcount; i++) {
    lua_rawgeti(L, tbl, (int)i);
    if (lua_istable(L, -1)) {
      int t = lua_gettop(L);
      Voxel v{0,0,0,255,255,255,100};
      v.x = (float)getNum(L, t, "x", 0);
      v.y = (float)getNum(L, t, "y", 0);
      v.z = (float)getNum(L, t, "z", 0);
      lua_getfield(L, t, "color");
      if (lua_istable(L, -1)) {
        v.r = (unsigned char)getFieldInteger(L, -1, "r", 255);
        v.g = (unsigned char)getFieldInteger(L, -1, "g", 255);
        v.b = (unsigned char)getFieldInteger(L, -1, "b", 255);
        v.a = (unsigned char)getFieldInteger(L, -1, "a", 100);
      }
      lua_pop(L, 1);
      // Ghost alpha: 50% of original
      v.a = (unsigned char)std::max(1, (int)(v.a * 0.5f));
      pvs.push_back(v);
    }
    lua_pop(L, 1);
  }
  lua_pop(L, 1);
  return pvs;
}

// Neighbor offsets for interior face culling (shared across renderers)
static const int PREVIEW_NEIGH[6][3] = {
  {0,0, 1},  // front: +Z
  {0,0,-1},  // back:  -Z
  {1,0, 0},  // right: +X
  {-1,0,0},  // left:  -X
  {0,1, 0},  // top:   +Y
  {0,-1,0}   // bottom:-Y
};

static int l_render_basic(lua_State* L){
  if(!lua_istable(L,1)||!lua_istable(L,2)){
    lua_pushnil(L); lua_pushstring(L,"expected (voxels, params)"); return 2;
  }
  int width  =(int)getNum(L,2,"width",200);
  int height =(int)getNum(L,2,"height",200);

  // Accept both "scale" and legacy "scaleLevel"
  float scale=(float)getNum(L,2,"scale",-12345.0);
  if (scale < 0) {
    scale = (float)getNum(L,2,"scaleLevel",1.0);
  }
  if (scale <= 0.f) scale = 1.f;

  // Mesh-mode flag (flat, unshaded; interior faces culled)
  bool meshMode = false;
  lua_getfield(L,2,"mesh");
  if (lua_isboolean(L,-1)) meshMode = lua_toboolean(L,-1)!=0;
  lua_pop(L,1);
  if (!meshMode) {
    lua_getfield(L,2,"meshMode");
    if (lua_isboolean(L,-1)) meshMode = lua_toboolean(L,-1)!=0;
    lua_pop(L,1);
  }

  float xRotDeg=(float)getNum(L,2,"xRotation",0.0);
  float yRotDeg=(float)getNum(L,2,"yRotation",0.0);
  float zRotDeg=(float)getNum(L,2,"zRotation",0.0);
  float shadeIntensity=(float)getNum(L,2,"basicShadeIntensity",100.0);
  float lightIntensity=(float)getNum(L,2,"basicLightIntensity",0.0);
  float fovDeg=(float)getNum(L,2,"fovDegrees",0.0);
    float viewportCutoffMultiplier = (float)getNum(L,2,"viewportCutoffMultiplier",2.0);
    if (viewportCutoffMultiplier <= 0.f) viewportCutoffMultiplier = 2.0f;
  lua_getfield(L,2,"orthogonal");
  bool orthogonal = lua_toboolean(L,-1)!=0; lua_pop(L,1);
  // perspectiveScaleRef ("middle" | "front" | "back")
  std::string perspRef = "middle";
  lua_getfield(L,2,"perspectiveScaleRef");
  if (lua_isstring(L,-1)) {
    const char* pr = lua_tostring(L,-1);
    if (pr && *pr) perspRef = pr;
  }
  lua_pop(L,1);
  // perspectiveIntensity: camera distance multiplier (default 1.0)
  float perspIntensity = (float)getNum(L,2,"perspectiveIntensity",1.0);
  if (perspIntensity <= 0.f) perspIntensity = 1.0f;
  // backgroundColor
  unsigned char bgR=0,bgG=0,bgB=0,bgA=0;
  lua_getfield(L,2,"backgroundColor");
  if(lua_istable(L,-1)){
    bgR=(unsigned char)getFieldInteger(L,-1,"r",0);
    bgG=(unsigned char)getFieldInteger(L,-1,"g",0);
    bgB=(unsigned char)getFieldInteger(L,-1,"b",0);
    bgA=(unsigned char)getFieldInteger(L,-1,"a",0);
  }
  lua_pop(L,1);

  // Edit mode: build ID map alongside pixel buffer
  bool editMode = false;
  lua_getfield(L,2,"editMode");
  if (lua_isboolean(L,-1)) editMode = lua_toboolean(L,-1)!=0;
  lua_pop(L,1);

  size_t count=lua_rawlen(L,1);
  lua_createtable(L,0,3);
  if(count==0){
    lua_pushinteger(L,width); lua_setfield(L,-2,"width");
    lua_pushinteger(L,height); lua_setfield(L,-2,"height");
    lua_pushlstring(L,"",0); lua_setfield(L,-2,"pixels");
    return 1;
  }
  std::vector<Voxel> voxels;
  voxels.reserve(count);
  float minX=1e9f,minY=1e9f,minZ=1e9f;
  float maxX=-1e9f,maxY=-1e9f,maxZ=-1e9f;
  for(size_t i=1;i<=count;i++){
    lua_rawgeti(L,1,(int)i);
    if(lua_istable(L,-1)){
      int tblIdx = lua_gettop(L);
      Voxel v{0,0,0,255,255,255,255};
      // detect numeric form
      lua_rawgeti(L,tblIdx,1);
      bool numeric = lua_isnumber(L,-1)!=0;
      lua_pop(L,1);
      if(numeric){
        lua_rawgeti(L,tblIdx,1); v.x=(float)lua_tonumber(L,-1); lua_pop(L,1);
        lua_rawgeti(L,tblIdx,2); v.y=(float)lua_tonumber(L,-1); lua_pop(L,1);
        lua_rawgeti(L,tblIdx,3); v.z=(float)lua_tonumber(L,-1); lua_pop(L,1);
        lua_rawgeti(L,tblIdx,4); v.r=(unsigned char)lua_tointeger(L,-1); lua_pop(L,1);
        lua_rawgeti(L,tblIdx,5); v.g=(unsigned char)lua_tointeger(L,-1); lua_pop(L,1);
        lua_rawgeti(L,tblIdx,6); v.b=(unsigned char)lua_tointeger(L,-1); lua_pop(L,1);
        lua_rawgeti(L,tblIdx,7); v.a=(unsigned char)lua_tointeger(L,-1); lua_pop(L,1);
      } else {
        v.x=(float)getNum(L,tblIdx,"x",0);
        v.y=(float)getNum(L,tblIdx,"y",0);
        v.z=(float)getNum(L,tblIdx,"z",0);
        lua_getfield(L,tblIdx,"color");
        if(lua_istable(L,-1)){
          v.r=(unsigned char)getFieldInteger(L,-1,"r",255);
          v.g=(unsigned char)getFieldInteger(L,-1,"g",255);
          v.b=(unsigned char)getFieldInteger(L,-1,"b",255);
          v.a=(unsigned char)getFieldInteger(L,-1,"a",255);
        }
        lua_pop(L,1);
      }
      if(v.x<minX)minX=v.x; if(v.x>maxX)maxX=v.x;
      if(v.y<minY)minY=v.y; if(v.y>maxY)maxY=v.y;
      if(v.z<minZ)minZ=v.z; if(v.z>maxZ)maxZ=v.z;
      voxels.push_back(v);
    }
    lua_pop(L,1);
  }

  // Build occupancy map for mesh mode (string key "x,y,z")
  std::unordered_set<std::string> occ;
  if (meshMode) {
    occ.reserve(voxels.size()*2);
    for (auto &v : voxels) {
      int ix = (int)std::lround(v.x);
      int iy = (int)std::lround(v.y);
      int iz = (int)std::lround(v.z);
      occ.emplace(std::to_string(ix)+","+std::to_string(iy)+","+std::to_string(iz));
    }
  }

  // Read custom middlePoint from params if provided, otherwise calculate from bounds
  float midX, midY, midZ;
  lua_getfield(L, 2, "middlePoint");
  if (lua_istable(L, -1)) {
      midX = (float)getNum(L, -1, "x", 0.5f * (minX + maxX));
      midY = (float)getNum(L, -1, "y", 0.5f * (minY + maxY));
      midZ = (float)getNum(L, -1, "z", 0.5f * (minZ + maxZ));
  } else {
      midX = 0.5f*(minX+maxX);
      midY = 0.5f*(minY+maxY);
      midZ = 0.5f*(minZ+maxZ);
  }
  lua_pop(L, 1);
  // Use globalBounds (if provided) for camera calculations
  CameraBounds camB = computeCameraBounds(L, 2, minX, maxX, minY, maxY, minZ, maxZ, midX, midY, midZ);
  float sizeX=camB.sizeX;
  float sizeY=camB.sizeY;
  float sizeZ=camB.sizeZ;
  float maxDim=camB.maxDim;
  float effectiveSpan=camB.effectiveSpan;

  float RX=xRotDeg*(float)PI/180.f;
  float RY=yRotDeg*(float)PI/180.f;
  float RZ=zRotDeg*(float)PI/180.f;
  float cx=std::cos(RX), sx=std::sin(RX);
  float cy=std::cos(RY), sy=std::sin(RY);
  float cz=std::cos(RZ), sz=std::sin(RZ);

  // Pre-computed 3x3 rotation matrix (Rz * Ry * Rx) — hoisted out of voxel loop
  float R00 = cy*cz,             R01 = sx*sy*cz - cx*sz,  R02 = cx*sy*cz + sx*sz;
  float R10 = cy*sz,             R11 = sx*sy*sz + cx*cz,  R12 = cx*sy*sz - sx*cz;
  float R20 = -sy,               R21 = sx*cy,             R22 = cx*cy;

  // ----------------------------------------------------------------------------
  // PERSPECTIVE & ORTHOGRAPHIC CAMERA (MATCH previewRenderer.lua)
  //   - Stronger FOV warping: camera distance shrinks non‑linearly as FOV increases
  //   - Reference depth scaling: chosen perspectiveScaleRef ("front","middle","back")
  //     stays closest to the user scale while allowing other depths to warp.
  //   - Orthographic path simplified (no template voxelSize distortion).
  // ----------------------------------------------------------------------------
  bool perspective = (!orthogonal && fovDeg > 0.f);
  float focalLength = 0.f;
  float camDist;
  if (perspective) {
    // Clamp FOV and compute warp curve -> amplified factor
    fovDeg = std::max(5.f, std::min(75.f, fovDeg));
    float warpT = (fovDeg - 5.f) / (75.f - 5.f);
    if (warpT < 0.f) warpT = 0.f; if (warpT > 1.f) warpT = 1.f;
    float amplified = std::pow(warpT, 1.f/3.f);
    const float BASE_NEAR = 1.2f;
    const float FAR_EXTRA = 45.f;
    camDist = effectiveSpan * (BASE_NEAR + (1.f - amplified)*(1.f - amplified) * FAR_EXTRA);
    camDist *= perspIntensity;  // Apply perspective intensity multiplier
    float fovRad = fovDeg * (float)PI / 180.f;
    focalLength = (height / 2.f) / std::tan(fovRad / 2.f);
  } else {
    camDist = effectiveSpan * 5.f;
  }

  // Compute voxelSize
  float voxelSize = scale;
  {
    float maxAllowedPixels = std::min(width, height) * viewportCutoffMultiplier;
    if (perspective && focalLength > 1e-6f && maxDim > 0.f) {
      // Rotate AABB corners to derive front/back depths (match basic renderer)
      float RX = xRotDeg*(float)PI/180.f;
      float RY = yRotDeg*(float)PI/180.f;
      float RZ = zRotDeg*(float)PI/180.f;
      float rcx=std::cos(RX), rsx=std::sin(RX);
      float rcy=std::cos(RY), rsy=std::sin(RY);
      float rcz=std::cos(RZ), rsz=std::sin(RZ);
      float zMinRot =  1e9f;
      float zMaxRot = -1e9f;
      for (int ix=0; ix<2; ++ix){
        float cxv = (ix==0)?camB.camMinX:camB.camMaxX;
        for (int iy=0; iy<2; ++iy){
          float cyv = (iy==0)?camB.camMinY:camB.camMaxY;
          for (int iz=0; iz<2; ++iz){
            float czv = (iz==0)?camB.camMinZ:camB.camMaxZ;
            float X = cxv - midX;
            float Y = cyv - midY;
            float Z = czv - midZ;
            { float Y2=Y*rcx - Z*rsx; float Z2=Y*rsx + Z*rcx; Y=Y2; Z=Z2; }
            { float X2=X*rcy + Z*rsy; float Z3=-X*rsy + Z*rcy; X=X2; Z=Z3; }
            { float X3=X*rcz - Y*rsz; float Y3=X*rsz + Y*rcz; X=X3; Y=Y3; }
            float worldZ = Z + midZ;
            if (worldZ < zMinRot) zMinRot = worldZ;
            if (worldZ > zMaxRot) zMaxRot = worldZ;
          }
        }
      }
      float cameraZ = midZ + camDist;
      float depthBack  = std::max(0.001f, cameraZ - zMinRot); // farthest
      float depthFront = std::max(0.001f, cameraZ - zMaxRot); // nearest
      float depthMiddle= std::max(0.001f, camDist);        // model center
      float depthRef = depthMiddle;
      if (perspRef == "front" || perspRef == "Front") depthRef = depthFront;
      else if (perspRef == "back" || perspRef == "Back") depthRef = depthBack;
      voxelSize = scale * (depthRef / focalLength);
      if (voxelSize * effectiveSpan > maxAllowedPixels) voxelSize = maxAllowedPixels / effectiveSpan;
      if (voxelSize <= 0.f) voxelSize = 1.f;
    } else {
      if (voxelSize < 1.f) voxelSize = 1.f;
      if (voxelSize * effectiveSpan > maxAllowedPixels && effectiveSpan > 0.f)
        voxelSize *= (maxAllowedPixels / (voxelSize * effectiveSpan));
    }
  }
  
  // Orthographic template only if voxelSize not degenerate (avoid artifacts)
  // FIX (Basic Ortho Speckle):
  // Previously we used a cached orthographic template (tmpl) that:
  //   1. Quantized angles to whole degrees.
  //   2. Dropped per-vertex rotated Z (depth) variation.
  //   3. Gave every visible face in a voxel identical avgDepth.
  // Result: large numbers of ties in painter's sort => unstable overdraw
  // producing single‑pixel "speckles".
  //
  // To restore correctness we disable the template optimization here and
  // always execute the per‑voxel vertex rotation path (tmpl = nullptr).
  const RotatedFaceTemplate* tmpl = nullptr; // forced off for correctness

  // Pre-rotate face normals using the combined rotation matrix (avoids per-face rotateNormal)
  float rotNormals[6][3];
  for (int f = 0; f < 6; f++) {
    float lnx = LOCAL_FACE_NORMALS[f][0];
    float lny = LOCAL_FACE_NORMALS[f][1];
    float lnz = LOCAL_FACE_NORMALS[f][2];
    rotNormals[f][0] = R00*lnx + R01*lny + R02*lnz;
    rotNormals[f][1] = R10*lnx + R11*lny + R12*lnz;
    rotNormals[f][2] = R20*lnx + R21*lny + R22*lnz;
  }

  // Pre-rotate unit vertices scaled by voxelSize (avoids per-vertex rotateNormal)
  float rotVerts[8][3];
  for (int v = 0; v < 8; v++) {
    float lx = UNIT_VERTS[v][0] * voxelSize;
    float ly = UNIT_VERTS[v][1] * voxelSize;
    float lz = UNIT_VERTS[v][2] * voxelSize;
    rotVerts[v][0] = R00*lx + R01*ly + R02*lz;
    rotVerts[v][1] = R10*lx + R11*ly + R12*lz;
    rotVerts[v][2] = R20*lx + R21*ly + R22*lz;
  }

  std::vector<FacePoly> polys;

  // Use slightly negative threshold to ensure edge-on faces are consistently visible
  // This prevents jagged edges at certain rotation angles where faces are nearly perpendicular
  float threshold = -0.001f;

  // Neighbor offsets to cull interior faces (mesh mode)
  struct NOff { int dx,dy,dz; };
  static const NOff neigh[6] = {
    {0,0, 1},  // front
    {0,0,-1},  // back
    {1,0, 0},  // right
    {-1,0,0},  // left
    {0,1, 0},  // top
    {0,-1,0}   // bottom
  };

  // --------------------------------------------------------------------------
  // UNIFORM VIEW VECTOR FOR BASIC SHADING (anti-speckle fix)
  // --------------------------------------------------------------------------
  // Problem:
  //   When using interactive (relative / absolute) rotation the same Euler
  //   orientation can be reached through incremental matrix multiplications.
  //   Because the previous implementation used a PER-VOXEL view vector
  //   (camera -> voxel center) both face visibility and brightness varied
  //   slightly across a large, nominally flat face (e.g. the top plane).
  //   Small numeric differences around the visibility threshold + painter
  //   ordering produced isolated darker / lighter "dot" artifacts.
  //
  // Solution:
  //   For BASIC shading we want a stylized *uniform* face brightness.
  //   We keep per-voxel view vector ONLY for face visibility in perspective
  //   (so extreme parallax still culls correctly), but we compute a single
  //   global normalized camera direction (camera -> model center) and use
  //   that for:
  //     1. Face visibility threshold test (optional toggle – enabled here)
  //     2. Brightness calculation (always)
  //
  //   This removes sub‑voxel variation and stabilizes ordering across
  //   different rotation interaction paths.
  //
  //   If in the future nuanced per‑voxel falloff is desired for BASIC mode,
  //   guard it behind a param switch (not needed now).
  // --------------------------------------------------------------------------
  float gvx = 0.f, gvy = 0.f, gvz = 1.f; // default (orthographic or fallback)
  if (perspective) {
    gvx = camDist != 0.f ? (midX - midX) : 0.f; // 0
    gvy = camDist != 0.f ? (midY - midY) : 0.f; // 0
    gvz = (midZ + camDist) - midZ;              // camDist
    float gmag = std::sqrt(gvx*gvx + gvy*gvy + gvz*gvz);
    if (gmag > 1e-6f) { gvx /= gmag; gvy /= gmag; gvz /= gmag; }
  }

  // OpenMP thread-local face polygon buffers (avoid mutex on push_back)
  int maxThreads = omp_get_max_threads();
  std::vector<std::vector<FacePoly>> thread_polys(maxThreads);
  {
    size_t perThread = voxels.size() * 6 / maxThreads + 256;
    for (auto& tp : thread_polys) tp.reserve(perThread);
  }

  float screenCX = width * 0.5f;
  float screenCY = height * 0.5f;
  float camX = midX, camY = midY, camZ = midZ + camDist;

  #pragma omp parallel for schedule(dynamic, 256)
  for(size_t vi=0; vi<voxels.size(); vi++){
    const auto &v = voxels[vi];
    float x=v.x-midX, y=v.y-midY, z=v.z-midZ;
    // rotate center using pre-computed rotation matrix
    float rx = R00*x + R01*y + R02*z;
    float ry = R10*x + R11*y + R12*z;
    float rz = R20*x + R21*y + R22*z;
    float worldX = rx+midX;
    float worldY = ry+midY;
    float worldZ = rz+midZ;

    // Per-voxel view vector retained only for perspective culling logic
    float vxv=gvx, vyv=gvy, vzv=gvz;
    if (perspective) {
      // Use per-voxel vector for face visibility (wider FOV correctness),
      // but brightness will use the global (gvx,gvy,gvz) vector.
      float pvx = camX - worldX;
      float pvy = camY - worldY;
      float pvz = camZ - worldZ;
      float pmag = std::sqrt(pvx*pvx + pvy*pvy + pvz*pvz);
      if (pmag > 1e-5f) { pvx/=pmag; pvy/=pmag; pvz/=pmag; }
      vxv = pvx; vyv = pvy; vzv = pvz;
    }

    int tid = omp_get_thread_num();

    for(int f=0;f<6;f++){
      // Mesh mode: interior face culling via neighbor occupancy
      if (meshMode) {
        // faces order: 0=front,1=back,2=right,3=left,4=top,5=bottom
        int ix = (int)std::lround(v.x);
        int iy = (int)std::lround(v.y);
        int iz = (int)std::lround(v.z);
        int nnx = ix + neigh[f].dx;
        int nny = iy + neigh[f].dy;
        int nnz = iz + neigh[f].dz;
        std::string key = std::to_string(nnx)+","+std::to_string(nny)+","+std::to_string(nnz);
        if (occ.find(key) != occ.end()) {
          continue; // interior, skip
        }
      }

      // Use pre-rotated face normals (hoisted out of loop)
      float fnx = rotNormals[f][0];
      float fny = rotNormals[f][1];
      float fnz = rotNormals[f][2];

      // For consistency, evaluate visibility against *global* view dir in basic mode
      // so borderline faces don't flicker between interaction paths.
      float visDot = fnx*gvx + fny*gvy + fnz*gvz;
      if (visDot <= threshold) continue;

      // Brightness uses global view vector (gvx/gvy/gvz) for uniformity
      float bright = 1.0f;
      if (!meshMode) {
        float dot = visDot;
        bright = basicBrightness(dot,lightIntensity,shadeIntensity);
      }

      FacePoly poly{};
      poly.r=(unsigned char)std::round(std::min(255.f,v.r*bright));
      poly.g=(unsigned char)std::round(std::min(255.f,v.g*bright));
      poly.b=(unsigned char)std::round(std::min(255.f,v.b*bright));
      poly.a=v.a;
      // Store tiebreaker info for stable sorting
      poly.voxelX = v.x;
      poly.voxelY = v.y;
      poly.voxelZ = v.z;
      poly.faceIdx = f;
      poly.voxelId = (uint32_t)(vi + 1); // 1-based to match Lua indexing
      float avgDepth=0.f;
      // Unified per-vertex path (perspective OR orthographic) to preserve
      // subtle per-vertex depth differences and deterministic order.
      {
        bool usePerspective = perspective;
        for(int i=0;i<4;i++){
          int vidx = FACE_IDX[f][i]-1;
          // Use pre-rotated vertices (hoisted out of loop)
          float lx = rotVerts[vidx][0];
          float ly = rotVerts[vidx][1];
          float lz = rotVerts[vidx][2];
          float wx = worldX*voxelSize + lx;
          float wy = worldY*voxelSize + ly;
          float wzLocal = worldZ + (lz/voxelSize);

          float depth;
          float s = 1.0f;
          if(usePerspective){
            depth = (camZ - wzLocal);
            if(depth < 0.001f) depth = 0.001f;
            s = (focalLength > 0.0f) ? (focalLength / depth) : 1.0f;
          } else {
            // Orthographic: retain subtle per-vertex depth; add tiny bias along normal
            depth = (camZ - wzLocal) + fnz * 0.001f;
          }

          poly.x[i] = screenCX + (wx - midX*voxelSize)*s;
          poly.y[i] = screenCY + (wy - midY*voxelSize)*s;
          avgDepth += depth;
        }
        avgDepth /= 4.f;
      }
      poly.depth=avgDepth;
      thread_polys[tid].push_back(poly);
    }
  } // end omp parallel for

  // Merge thread-local buffers into polys
  {
    size_t totalPolys = 0;
    for (const auto& tp : thread_polys) totalPolys += tp.size();
    polys.reserve(totalPolys);
    for (auto& tp : thread_polys) {
      polys.insert(polys.end(), std::make_move_iterator(tp.begin()), std::make_move_iterator(tp.end()));
      tp.clear();
      tp.shrink_to_fit();
    }
  }

  // ---------------------------------------------------------------------------
  // Preview voxels: ghost style (flat color, reduced alpha, no shading, no ID map)
  // ---------------------------------------------------------------------------
  auto previewVoxels = readPreviewVoxels(L, 2);
  if (!previewVoxels.empty()) {
    // Build combined occupancy (model + preview) for preview interior culling
    std::unordered_set<std::string> combinedOcc;
    combinedOcc.reserve((voxels.size() + previewVoxels.size()) * 2);
    for (auto &v : voxels) {
      int ix = (int)std::lround(v.x);
      int iy = (int)std::lround(v.y);
      int iz = (int)std::lround(v.z);
      combinedOcc.emplace(std::to_string(ix)+","+std::to_string(iy)+","+std::to_string(iz));
    }
    for (auto &v : previewVoxels) {
      int ix = (int)std::lround(v.x);
      int iy = (int)std::lround(v.y);
      int iz = (int)std::lround(v.z);
      combinedOcc.emplace(std::to_string(ix)+","+std::to_string(iy)+","+std::to_string(iz));
    }

    polys.reserve(polys.size() + previewVoxels.size() * 3);
    for (size_t vi = 0; vi < previewVoxels.size(); vi++) {
      const auto &v = previewVoxels[vi];
      float x = v.x - midX, y = v.y - midY, z = v.z - midZ;
      float rx = R00*x + R01*y + R02*z;
      float ry = R10*x + R11*y + R12*z;
      float rz = R20*x + R21*y + R22*z;
      float worldX = rx + midX, worldY = ry + midY, worldZ = rz + midZ;

      for (int f = 0; f < 6; f++) {
        // Interior face culling against combined occupancy
        {
          int ix = (int)std::lround(v.x);
          int iy = (int)std::lround(v.y);
          int iz = (int)std::lround(v.z);
          int nnx = ix + PREVIEW_NEIGH[f][0];
          int nny = iy + PREVIEW_NEIGH[f][1];
          int nnz = iz + PREVIEW_NEIGH[f][2];
          std::string key = std::to_string(nnx)+","+std::to_string(nny)+","+std::to_string(nnz);
          if (combinedOcc.find(key) != combinedOcc.end()) continue;
        }

        float fnx = rotNormals[f][0];
        float fny = rotNormals[f][1];
        float fnz = rotNormals[f][2];
        float visDot = fnx*gvx + fny*gvy + fnz*gvz;
        if (visDot <= threshold) continue;

        // Ghost: flat color, no brightness modulation
        FacePoly poly{};
        poly.r = v.r;
        poly.g = v.g;
        poly.b = v.b;
        poly.a = v.a; // already halved in readPreviewVoxels
        poly.voxelX = v.x;
        poly.voxelY = v.y;
        poly.voxelZ = v.z;
        poly.faceIdx = f;
        poly.voxelId = 0; // skip ID map for preview
        float avgDepth = 0.f;
        for (int i = 0; i < 4; i++) {
          int vidx = FACE_IDX[f][i] - 1;
          float lx = rotVerts[vidx][0];
          float ly = rotVerts[vidx][1];
          float lz = rotVerts[vidx][2];
          float wx = worldX*voxelSize + lx;
          float wy = worldY*voxelSize + ly;
          float wzLocal = worldZ + (lz / voxelSize);
          float depth;
          float s = 1.0f;
          if (perspective) {
            depth = (camZ - wzLocal);
            if (depth < 0.001f) depth = 0.001f;
            s = (focalLength > 0.0f) ? (focalLength / depth) : 1.0f;
          } else {
            depth = (camZ - wzLocal) + fnz * 0.001f;
          }
          poly.x[i] = screenCX + (wx - midX*voxelSize)*s;
          poly.y[i] = screenCY + (wy - midY*voxelSize)*s;
          avgDepth += depth;
        }
        poly.depth = avgDepth / 4.f;
        polys.push_back(poly);
      }
    }
  }

  // Stable sort: primary by depth (far to near), secondary by voxel position + face for tiebreaker
  std::sort(polys.begin(),polys.end(),[](const FacePoly&a,const FacePoly&b){
    constexpr float DEPTH_EPS = 0.0001f;
    if (std::abs(a.depth - b.depth) > DEPTH_EPS) {
      return a.depth > b.depth; // far to near
    }
    // Tiebreaker: consistent ordering by voxel position, then face index
    if (a.voxelZ != b.voxelZ) return a.voxelZ < b.voxelZ;
    if (a.voxelY != b.voxelY) return a.voxelY < b.voxelY;
    if (a.voxelX != b.voxelX) return a.voxelX < b.voxelX;
    return a.faceIdx < b.faceIdx;
  });

  std::vector<unsigned char> buffer((size_t)width*height*4,0);
  for(int y=0;y<height;y++){
    for(int x=0;x<width;x++){
      size_t off=((size_t)y*width + x)*4;
      buffer[off+0]=bgR; buffer[off+1]=bgG; buffer[off+2]=bgB; buffer[off+3]=bgA;
    }
  }
  std::vector<uint32_t> idMapBuf;
  std::vector<uint32_t>* idMapPtr = nullptr;
  if (editMode) {
    idMapBuf.resize((size_t)width*height, 0);
    idMapPtr = &idMapBuf;
  }
  for(auto &p: polys) rasterQuad(p,width,height,buffer,idMapPtr);

  lua_pushinteger(L,width); lua_setfield(L,-2,"width");
  lua_pushinteger(L,height); lua_setfield(L,-2,"height");
  lua_pushlstring(L,(const char*)buffer.data(), buffer.size());
  lua_setfield(L,-2,"pixels");
  if (editMode) {
    lua_pushlstring(L,(const char*)idMapBuf.data(), idMapBuf.size()*sizeof(uint32_t));
    lua_setfield(L,-2,"editIdMap");
  }
  return 1;
}
} // anonymous namespace

// ============================================================================
// Host Helpers Implementation
// ============================================================================

struct ModelContext {
    std::unordered_map<long long, Voxel*> voxel_map;
    int minX, minY, minZ;
    int sizeX, sizeY, sizeZ;
};

static long long pack_coords(int x, int y, int z) {
    return ((long long)(x + 1000000) << 40) | ((long long)(y + 1000000) << 20) | (z + 1000000);
}

extern "C" {

int native_model_get_size(void* model, int* out_x, int* out_y, int* out_z) {
    if (!model) return 0;
    ModelContext* ctx = (ModelContext*)model;
    if (out_x) *out_x = ctx->sizeX;
    if (out_y) *out_y = ctx->sizeY;
    if (out_z) *out_z = ctx->sizeZ;
    return 1;
}

int native_model_get_voxel(void* model, int x, int y, int z, unsigned char* out_rgba) {
    if (!model) return 0;
    ModelContext* ctx = (ModelContext*)model;
    long long key = pack_coords(x, y, z);
    auto it = ctx->voxel_map.find(key);
    if (it != ctx->voxel_map.end()) {
        Voxel* v = it->second;
        out_rgba[0] = v->r;
        out_rgba[1] = v->g;
        out_rgba[2] = v->b;
        out_rgba[3] = v->a;
        return 1;
    }
    return 0;
}

int native_model_is_visible(void* model, int x, int y, int z) {
    if (!model) return 0;
    ModelContext* ctx = (ModelContext*)model;
    long long key = pack_coords(x, y, z);
    return ctx->voxel_map.find(key) != ctx->voxel_map.end();
}

int native_surface_set_pixel(void* surface, int x, int y, unsigned char r, unsigned char g, unsigned char b, unsigned char a) {
    // Not implemented for now (rendering is done via rasterQuad)
    return 0;
}

int native_surface_get_size(void* surface, int* out_w, int* out_h) {
    return 0;
}

float asev_dot3(const float* a, const float* b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

void asev_normalize3(float* v) {
    float len = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    if (len > 1e-6f) { v[0]/=len; v[1]/=len; v[2]/=len; }
}

void asev_cross3(const float* a, const float* b, float* out) {
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

} // extern "C"

// Debug function to try loading a specific DLL and report results
static int l_try_load_shader_dll(lua_State* L) {
    const char* dll_path = luaL_checkstring(L, 1);
    
    lua_createtable(L, 0, 10);
    
    lua_pushstring(L, dll_path);
    lua_setfield(L, -2, "dll_path");
    
#ifdef _WIN32
    // Check if file exists
    DWORD attrs = GetFileAttributesA(dll_path);
    lua_pushboolean(L, attrs != INVALID_FILE_ATTRIBUTES);
    lua_setfield(L, -2, "file_exists");
    
    if (attrs == INVALID_FILE_ATTRIBUTES) {
        lua_pushinteger(L, GetLastError());
        lua_setfield(L, -2, "file_error");
        return 1;
    }
    
    // Try to load the DLL
    HMODULE handle = LoadLibraryA(dll_path);
    if (!handle) {
        DWORD err = GetLastError();
        lua_pushboolean(L, false);
        lua_setfield(L, -2, "load_success");
        lua_pushinteger(L, err);
        lua_setfield(L, -2, "load_error");
        
        // Common error codes
        const char* errMsg = "unknown";
        if (err == 126) errMsg = "ERROR_MOD_NOT_FOUND (missing dependency)";
        else if (err == 127) errMsg = "ERROR_PROC_NOT_FOUND";
        else if (err == 193) errMsg = "ERROR_BAD_EXE_FORMAT (32/64 bit mismatch?)";
        else if (err == 14001) errMsg = "ERROR_SXS_CANT_GEN_ACTCTX (manifest issue)";
        lua_pushstring(L, errMsg);
        lua_setfield(L, -2, "load_error_msg");
        return 1;
    }
    
    lua_pushboolean(L, true);
    lua_setfield(L, -2, "load_success");
    
    // Try to get the export function
    typedef const native_shader_v1_t* (*get_v1_fn)(void);
    get_v1_fn get_v1 = (get_v1_fn)GetProcAddress(handle, "native_shader_get_v1");
    
    if (!get_v1) {
        DWORD err = GetLastError();
        lua_pushboolean(L, false);
        lua_setfield(L, -2, "has_export");
        lua_pushinteger(L, err);
        lua_setfield(L, -2, "export_error");
        FreeLibrary(handle);
        return 1;
    }
    
    lua_pushboolean(L, true);
    lua_setfield(L, -2, "has_export");
    
    // Call the function to get the interface
    const native_shader_v1_t* iface = get_v1();
    if (!iface) {
        lua_pushboolean(L, false);
        lua_setfield(L, -2, "has_interface");
        FreeLibrary(handle);
        return 1;
    }
    
    lua_pushboolean(L, true);
    lua_setfield(L, -2, "has_interface");
    
    // Get shader info
    const char* shader_id = iface->shader_id();
    const char* display_name = iface->display_name();
    native_version_t ver = iface->api_version();
    
    lua_pushstring(L, shader_id ? shader_id : "(null)");
    lua_setfield(L, -2, "shader_id");
    lua_pushstring(L, display_name ? display_name : "(null)");
    lua_setfield(L, -2, "display_name");
    
    char ver_str[32];
    snprintf(ver_str, sizeof(ver_str), "%d.%d", ver.major, ver.minor);
    lua_pushstring(L, ver_str);
    lua_setfield(L, -2, "api_version");
    
    lua_pushinteger(L, native_shader_loader::EXPECTED_MAJOR);
    lua_setfield(L, -2, "expected_major");
    
    lua_pushboolean(L, ver.major == native_shader_loader::EXPECTED_MAJOR);
    lua_setfield(L, -2, "version_ok");
    
    FreeLibrary(handle);
#else
    lua_pushstring(L, "linux - not implemented");
    lua_setfield(L, -2, "error");
#endif
    
    return 1;
}

// Debug function to return diagnostic info to Lua
static int l_scan_shaders_debug(lua_State* L) {
    const char* path = luaL_checkstring(L, 1);
    
    lua_createtable(L, 0, 10);
    
    lua_pushstring(L, path);
    lua_setfield(L, -2, "input_path");
    
#ifdef _WIN32
    lua_pushstring(L, "win32");
    lua_setfield(L, -2, "platform");
    
    // Test if directory exists using GetFileAttributesA
    DWORD attrs = GetFileAttributesA(path);
    lua_pushboolean(L, attrs != INVALID_FILE_ATTRIBUTES && (attrs & FILE_ATTRIBUTE_DIRECTORY));
    lua_setfield(L, -2, "dir_exists");
    
    if (attrs == INVALID_FILE_ATTRIBUTES) {
        lua_pushinteger(L, GetLastError());
        lua_setfield(L, -2, "dir_error");
    }
    
    // Try each pattern and report results
    const char* patterns[] = {
        "\\libnative_shader_*.dll",
        "\\native_shader_*.dll",
        "\\*.dll",  // Try listing ALL DLLs
        NULL
    };
    
    lua_createtable(L, 0, 3);
    for (int p = 0; patterns[p] != NULL; p++) {
        std::string search_path = std::string(path) + patterns[p];
        
        lua_createtable(L, 0, 4);
        lua_pushstring(L, search_path.c_str());
        lua_setfield(L, -2, "search_path");
        
        WIN32_FIND_DATAA findData;
        HANDLE hFind = FindFirstFileA(search_path.c_str(), &findData);
        
        if (hFind == INVALID_HANDLE_VALUE) {
            DWORD err = GetLastError();
            lua_pushboolean(L, false);
            lua_setfield(L, -2, "found");
            lua_pushinteger(L, err);
            lua_setfield(L, -2, "error_code");
            
            // Error code meaning
            const char* errMsg = "unknown";
            if (err == 2) errMsg = "ERROR_FILE_NOT_FOUND";
            else if (err == 3) errMsg = "ERROR_PATH_NOT_FOUND";
            else if (err == 5) errMsg = "ERROR_ACCESS_DENIED";
            lua_pushstring(L, errMsg);
            lua_setfield(L, -2, "error_msg");
        } else {
            lua_pushboolean(L, true);
            lua_setfield(L, -2, "found");
            
            // List files found
            lua_createtable(L, 0, 0);
            int fileIdx = 1;
            do {
                if (!(findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                    lua_pushstring(L, findData.cFileName);
                    lua_rawseti(L, -2, fileIdx++);
                }
            } while (FindNextFileA(hFind, &findData) && fileIdx <= 20);
            FindClose(hFind);
            
            lua_setfield(L, -2, "files");
        }
        
        lua_rawseti(L, -2, p + 1);
    }
    lua_setfield(L, -2, "pattern_tests");
#else
    lua_pushstring(L, "linux");
    lua_setfield(L, -2, "platform");
#endif
    
    return 1;
}

static int l_scan_shaders(lua_State* L) {
    const char* path = luaL_checkstring(L, 1);
    int count = native_shader_loader::scan_directory(path);
    
    // Create result table with count and shaders fields
    lua_createtable(L, 0, 2);
    
    // Add count field
    lua_pushinteger(L, count);
    lua_setfield(L, -2, "count");
    
    // Create shaders array
    lua_createtable(L, count, 0);
    for(int i=0; i<count; i++) {
        const char* id = native_shader_loader::get_shader_id(i);
        if(id) {
            const native_shader_v1_t* iface = native_shader_loader::get_shader_interface(id);
            if(iface) {
                lua_createtable(L, 0, 3);
                lua_pushstring(L, iface->shader_id()); lua_setfield(L, -2, "id");
                lua_pushstring(L, iface->display_name()); lua_setfield(L, -2, "name");
                
                native_version_t v = iface->api_version();
                char ver[32];
                snprintf(ver, sizeof(ver), "%d.%d", v.major, v.minor);
                lua_pushstring(L, ver); lua_setfield(L, -2, "version");
                
                lua_rawseti(L, -2, i+1);
            }
        }
    }
    lua_setfield(L, -2, "shaders");
    
    return 1;
}

// Unload all native shaders - call this before extension uninstall/update
static int l_unload_shaders(lua_State* L) {
    (void)L;  // Unused
    native_shader_loader::unload_all();
    return 0;
}

// ============================================================================
// Phase 2.3/3: Parse shaderStack config (lighting/fx arrays)
// Returns vector of ActiveShader structs ready for execution
// ============================================================================
struct ActiveShader {
    void* instance;
    const native_shader_v1_t* iface;
    std::string id;
    std::string category; // "lighting" or "fx"
    std::string colorInput; // "baseColor", "last", or shader outputName
    std::string outputName; // Name for this shader's output (for color routing)
    std::string addTo;      // Optional: add result to this output
    int localIndex;         // Index within category for auto-naming
};

// Debug flag for shader parameter logging (set via DEBUG_SHADER_PARAMS env or code)
static bool g_debug_shader_params = false;

static void parseShaderCategory(lua_State* L, int categoryTableIdx, 
                                 const char* categoryName,
                                 std::vector<ActiveShader>& out_shaders) {
    if (!lua_istable(L, categoryTableIdx)) return;
    
    size_t count = lua_rawlen(L, categoryTableIdx);
    
    if (g_debug_shader_params) {
        printf("[native] parseShaderCategory: %s with %zu entries\n", categoryName, count);
    }
    
    for (size_t i = 1; i <= count; i++) {
        lua_rawgeti(L, categoryTableIdx, (int)i);
        if (!lua_istable(L, -1)) { lua_pop(L, 1); continue; }
        
        // Check if enabled
        lua_getfield(L, -1, "enabled");
        bool enabled = lua_isnil(L, -1) || lua_toboolean(L, -1); // default true
        lua_pop(L, 1);
        
        if (!enabled) { lua_pop(L, 1); continue; }
        
        // Get shader ID - try both native_id and id
        lua_getfield(L, -1, "native_id");
        if (lua_isnil(L, -1)) {
            lua_pop(L, 1);
            lua_getfield(L, -1, "id");
        }
        const char* shader_id = lua_tostring(L, -1);
        lua_pop(L, 1);
        
        if (!shader_id) { lua_pop(L, 1); continue; }
        
        // Look up shader by full ID or by category-prefixed name
        std::string full_id = shader_id;
        const native_shader_v1_t* iface = native_shader_loader::get_shader_interface(shader_id);
        
        // Try with asevoxel. prefix if not found
        if (!iface) {
            std::string prefixed = std::string("asevoxel.") + shader_id;
            iface = native_shader_loader::get_shader_interface(prefixed.c_str());
            if (iface) full_id = prefixed;
        }
        
        // Try with pixelmatt. prefix if not found
        if (!iface) {
            std::string prefixed = std::string("pixelmatt.") + shader_id;
            iface = native_shader_loader::get_shader_interface(prefixed.c_str());
            if (iface) full_id = prefixed;
        }
        
        if (!iface) { 
            if (g_debug_shader_params) {
                printf("[native] Shader NOT FOUND: %s\n", shader_id);
            }
            lua_pop(L, 1); 
            continue; 
        }
        
        if (g_debug_shader_params) {
            printf("[native] Found shader: %s -> %s\n", shader_id, full_id.c_str());
        }
        
        // Create instance
        void* instance = native_shader_loader::create_shader_instance(full_id.c_str());
        if (!instance) { lua_pop(L, 1); continue; }
        
        // Parse params table and set on instance
        // NOTE: Native shaders expect ALL parameters as strings
        lua_getfield(L, -1, "params");
        if (lua_istable(L, -1)) {
            int params_set = 0;
            lua_pushnil(L);
            while (lua_next(L, -2) != 0) {
                const char* key = lua_tostring(L, -2);
                if (key) {
                    char str_value[256] = {0};
                    bool valid = false;
                    
                    if (lua_isstring(L, -1)) {
                        // String: pass directly
                        const char* s = lua_tostring(L, -1);
                        if (s) {
                            strncpy(str_value, s, sizeof(str_value) - 1);
                            valid = true;
                        }
                    } else if (lua_isnumber(L, -1)) {
                        // Number: convert to string
                        if (lua_isinteger(L, -1)) {
                            snprintf(str_value, sizeof(str_value), "%lld", (long long)lua_tointeger(L, -1));
                        } else {
                            snprintf(str_value, sizeof(str_value), "%g", lua_tonumber(L, -1));
                        }
                        valid = true;
                    } else if (lua_isboolean(L, -1)) {
                        // Boolean: "true" or "false"
                        strncpy(str_value, lua_toboolean(L, -1) ? "true" : "false", sizeof(str_value) - 1);
                        valid = true;
                    } else if (lua_istable(L, -1)) {
                        // Table: might be a color {r,g,b,a}
                        lua_getfield(L, -1, "r");
                        if (lua_isnumber(L, -1)) {
                            int r = (int)lua_tointeger(L, -1);
                            lua_pop(L, 1);
                            lua_getfield(L, -1, "g");
                            int g = lua_isnumber(L, -1) ? (int)lua_tointeger(L, -1) : 255;
                            lua_pop(L, 1);
                            lua_getfield(L, -1, "b");
                            int b = lua_isnumber(L, -1) ? (int)lua_tointeger(L, -1) : 255;
                            lua_pop(L, 1);
                            lua_getfield(L, -1, "a");
                            int a = lua_isnumber(L, -1) ? (int)lua_tointeger(L, -1) : 255;
                            lua_pop(L, 1);
                            // Format as "r,g,b,a" - matches shader parse_color() expectations
                            snprintf(str_value, sizeof(str_value), "%d,%d,%d,%d", r, g, b, a);
                            valid = true;
                        } else {
                            lua_pop(L, 1);
                        }
                    }
                    
                    if (valid) {
                        int result = iface->set_param(instance, key, str_value);
                        if (g_debug_shader_params) {
                            printf("[native]   param %s = \"%s\" -> %s\n", 
                                   key, str_value, result == 0 ? "OK" : "FAIL");
                        }
                        params_set++;
                    }
                }
                lua_pop(L, 1); // value, keep key for next iteration
            }
            if (g_debug_shader_params) {
                printf("[native] Shader %s: set %d params\n", full_id.c_str(), params_set);
            }
        }
        lua_pop(L, 1); // params
        
        // Parse colorInput (defaults to "last")
        std::string colorInput = "last";
        lua_getfield(L, -1, "colorInput");
        if (lua_isstring(L, -1)) {
            colorInput = lua_tostring(L, -1);
        }
        lua_pop(L, 1);
        
        // Parse outputName (optional)
        std::string outputName = "";
        lua_getfield(L, -1, "outputName");
        if (lua_isstring(L, -1)) {
            outputName = lua_tostring(L, -1);
        }
        lua_pop(L, 1);
        
        // Parse addTo (optional - for additive blending)
        std::string addTo = "";
        lua_getfield(L, -1, "addTo");
        if (lua_isstring(L, -1)) {
            addTo = lua_tostring(L, -1);
        }
        lua_pop(L, 1);
        
        // Count shaders in this category for auto-naming
        int localIndex = 0;
        for (const auto& s : out_shaders) {
            if (s.category == categoryName) localIndex++;
        }
        
        ActiveShader shader;
        shader.instance = instance;
        shader.iface = iface;
        shader.id = full_id;
        shader.category = categoryName;
        shader.colorInput = colorInput;
        shader.outputName = outputName;
        shader.addTo = addTo;
        shader.localIndex = localIndex;
        out_shaders.push_back(shader);
        
        if (g_debug_shader_params) {
            printf("[native] Shader %s: colorInput=%s outputName=%s addTo=%s\n", 
                   full_id.c_str(), colorInput.c_str(), outputName.c_str(), addTo.c_str());
        }
        
        lua_pop(L, 1); // shader entry table
    }
}

// ============================================================================
// Phase 3: Parse visibility data from params._visibilityData
// The Lua side passes pre-computed visibility as:
// _visibilityData = {
//   voxels = {
//     { x=0, y=0, z=0, faces = { 
//       { name="front", normal={0,0,1}, color={r,g,b,a} },
//       ...
//     }},
//     ...
//   }
// }
// ============================================================================
static void parseVisibilityData(lua_State* L, int paramsIdx,
                                 std::vector<native_face_data_t>& out_faces,
                                 const ModelContext* modelCtx) {
    lua_getfield(L, paramsIdx, "_visibilityData");
    if (!lua_istable(L, -1)) { lua_pop(L, 1); return; }
    
    lua_getfield(L, -1, "voxels");
    if (!lua_istable(L, -1)) { lua_pop(L, 2); return; }
    
    size_t voxelCount = lua_rawlen(L, -1);
    for (size_t vi = 1; vi <= voxelCount; vi++) {
        lua_rawgeti(L, -1, (int)vi);
        if (!lua_istable(L, -1)) { lua_pop(L, 1); continue; }
        
        int x = (int)getNum(L, -1, "x", 0);
        int y = (int)getNum(L, -1, "y", 0);
        int z = (int)getNum(L, -1, "z", 0);
        
        // Get voxel color from model context
        unsigned char vr = 255, vg = 255, vb = 255, va = 255;
        uint64_t key = pack_coords(x, y, z);
        auto it = modelCtx->voxel_map.find(key);
        if (it != modelCtx->voxel_map.end()) {
            vr = it->second->r;
            vg = it->second->g;
            vb = it->second->b;
            va = it->second->a;
        }
        
        lua_getfield(L, -1, "faces");
        if (!lua_istable(L, -1)) { lua_pop(L, 2); continue; }
        
        size_t faceCount = lua_rawlen(L, -1);
        for (size_t fi = 1; fi <= faceCount; fi++) {
            lua_rawgeti(L, -1, (int)fi);
            if (!lua_istable(L, -1)) { lua_pop(L, 1); continue; }
            
            native_face_data_t face;
            face.x = x;
            face.y = y;
            face.z = z;
            
            // Parse face name to get face_dir
            // Try both "name" (legacy) and "face" (new) field names
            lua_getfield(L, -1, "name");
            const char* faceName = lua_tostring(L, -1);
            lua_pop(L, 1);
            
            if (!faceName) {
                // Try "face" field (used by shader_stack.lua)
                lua_getfield(L, -1, "face");
                faceName = lua_tostring(L, -1);
                lua_pop(L, 1);
            }
            
            // Map face name to direction
            // API: 0=+x, 1=-x, 2=+y, 3=-y, 4=+z, 5=-z
            face.face_dir = 4; // default front (+z)
            if (faceName) {
                if (strcmp(faceName, "front") == 0) face.face_dir = 4;      // +Z
                else if (strcmp(faceName, "back") == 0) face.face_dir = 5;  // -Z
                else if (strcmp(faceName, "right") == 0) face.face_dir = 0; // +X
                else if (strcmp(faceName, "left") == 0) face.face_dir = 1;  // -X
                else if (strcmp(faceName, "top") == 0) face.face_dir = 2;   // +Y
                else if (strcmp(faceName, "bottom") == 0) face.face_dir = 3; // -Y
            }
            
            // Parse normal
            lua_getfield(L, -1, "normal");
            if (lua_istable(L, -1)) {
                // normal can be array {0,0,1} or table {x=0,y=0,z=1}
                lua_rawgeti(L, -1, 1);
                if (lua_isnumber(L, -1)) {
                    face.normal[0] = (float)lua_tonumber(L, -1);
                    lua_pop(L, 1);
                    lua_rawgeti(L, -1, 2); face.normal[1] = (float)lua_tonumber(L, -1); lua_pop(L, 1);
                    lua_rawgeti(L, -1, 3); face.normal[2] = (float)lua_tonumber(L, -1); lua_pop(L, 1);
                } else {
                    lua_pop(L, 1);
                    face.normal[0] = (float)getNum(L, -1, "x", 0);
                    face.normal[1] = (float)getNum(L, -1, "y", 0);
                    face.normal[2] = (float)getNum(L, -1, "z", 1);
                }
            } else {
                // Default normal based on face direction
                face.normal[0] = 0; face.normal[1] = 0; face.normal[2] = 0;
                switch(face.face_dir) {
                    case 0: face.normal[0] = 1; break;
                    case 1: face.normal[0] = -1; break;
                    case 2: face.normal[1] = 1; break;
                    case 3: face.normal[1] = -1; break;
                    case 4: face.normal[2] = 1; break;
                    case 5: face.normal[2] = -1; break;
                }
            }
            lua_pop(L, 1); // normal
            
            // Parse color (may override voxel color)
            lua_getfield(L, -1, "color");
            if (lua_istable(L, -1)) {
                face.r = (unsigned char)getFieldInteger(L, -1, "r", vr);
                face.g = (unsigned char)getFieldInteger(L, -1, "g", vg);
                face.b = (unsigned char)getFieldInteger(L, -1, "b", vb);
                face.a = (unsigned char)getFieldInteger(L, -1, "a", va);
            } else {
                face.r = vr; face.g = vg; face.b = vb; face.a = va;
            }
            lua_pop(L, 1); // color
            
            face.depth = 0.0f; // Will be computed during projection
            
            out_faces.push_back(face);
            lua_pop(L, 1); // face entry
        }
        lua_pop(L, 1); // faces
        lua_pop(L, 1); // voxel entry
    }
    lua_pop(L, 2); // voxels, _visibilityData
}

// ============================================================================
// render_shader_stack: New Phase 2.3/3 shader stack renderer
// Handles both old fxStack.modules[] and new shaderStack { lighting=[], fx=[] }
// ============================================================================
static int l_render_shader_stack(lua_State* L) {
    if (!lua_istable(L, 1) || !lua_istable(L, 2)) {
        lua_pushnil(L);
        lua_pushstring(L, "expected (voxels, params)");
        return 2;
    }
    
    // Parse rendering parameters
    int width = (int)getNum(L, 2, "width", 200);
    int height = (int)getNum(L, 2, "height", 200);
    float scale = (float)getNum(L, 2, "scale", -12345.0f);
    if (scale < 0) scale = (float)getNum(L, 2, "scaleLevel", 1.0f);
    if (scale <= 0) scale = 1.0f;
    float viewportCutoffMultiplier = (float)getNum(L, 2, "viewportCutoffMultiplier", 2.0f);
    if (viewportCutoffMultiplier <= 0.f) viewportCutoffMultiplier = 2.0f;
    float xRotDeg = (float)getNum(L, 2, "xRotation", 0.0f);
    float yRotDeg = (float)getNum(L, 2, "yRotation", 0.0f);
    float zRotDeg = (float)getNum(L, 2, "zRotation", 0.0f);
    float fovDeg = (float)getNum(L, 2, "fovDegrees", 0.0f);
    
    lua_getfield(L, 2, "orthogonal");
    bool orth = lua_toboolean(L, -1) != 0;
    lua_pop(L, 1);
    
    std::string perspRef = "middle";
    lua_getfield(L, 2, "perspectiveScaleRef");
    if (lua_isstring(L, -1)) perspRef = lua_tostring(L, -1);
    lua_pop(L, 1);
    // perspectiveIntensity: camera distance multiplier (default 1.0)
    float perspIntensity = (float)getNum(L, 2, "perspectiveIntensity", 1.0f);
    if (perspIntensity <= 0.f) perspIntensity = 1.0f;
    
    unsigned char bgR = 0, bgG = 0, bgB = 0, bgA = 0;
    lua_getfield(L, 2, "backgroundColor");
    if (lua_istable(L, -1)) {
        bgR = (unsigned char)getFieldInteger(L, -1, "r", 0);
        bgG = (unsigned char)getFieldInteger(L, -1, "g", 0);
        bgB = (unsigned char)getFieldInteger(L, -1, "b", 0);
        bgA = (unsigned char)getFieldInteger(L, -1, "a", 0);
    }
    lua_pop(L, 1);
    
    // Parse editMode for ID map output
    bool editMode = false;
    lua_getfield(L, 2, "editMode");
    if (lua_isboolean(L, -1)) editMode = lua_toboolean(L, -1) != 0;
    lua_pop(L, 1);
    
    // ========================================================================
    // Phase 2.3: Parse shader stack configuration
    // Support both:
    //   - New format: shaderStack = { lighting = {...}, fx = {...} }
    //   - Legacy format: fxStack = { modules = {...} }
    // ========================================================================
    std::vector<ActiveShader> active_shaders;
    
    // Try new shaderStack format first
    lua_getfield(L, 2, "shaderStack");
    if (lua_istable(L, -1)) {
        // Parse lighting shaders
        lua_getfield(L, -1, "lighting");
        if (lua_istable(L, -1)) {
            parseShaderCategory(L, lua_gettop(L), "lighting", active_shaders);
        }
        lua_pop(L, 1);
        
        // Parse FX shaders
        lua_getfield(L, -1, "fx");
        if (lua_istable(L, -1)) {
            parseShaderCategory(L, lua_gettop(L), "fx", active_shaders);
        }
        lua_pop(L, 1);
    }
    lua_pop(L, 1); // shaderStack
    
    // Fallback to legacy fxStack format
    if (active_shaders.empty()) {
        lua_getfield(L, 2, "fxStack");
        if (lua_istable(L, -1)) {
            lua_getfield(L, -1, "modules");
            if (lua_istable(L, -1)) {
                parseShaderCategory(L, lua_gettop(L), "fx", active_shaders);
            }
            lua_pop(L, 1);
        }
        lua_pop(L, 1);
    }
    
    // ========================================================================
    // Parse voxel data
    // ========================================================================
    size_t count = lua_rawlen(L, 1);
    std::vector<Voxel> voxels;
    voxels.reserve(count);
    
    float minX = 1e9f, minY = 1e9f, minZ = 1e9f;
    float maxX = -1e9f, maxY = -1e9f, maxZ = -1e9f;
    
    ModelContext modelCtx;
    
    for (size_t i = 1; i <= count; i++) {
        lua_rawgeti(L, 1, (int)i);
        if (!lua_istable(L, -1)) { lua_pop(L, 1); continue; }
        
        int tblIdx = lua_gettop(L);
        Voxel v{0, 0, 0, 255, 255, 255, 255};
        
        lua_rawgeti(L, tblIdx, 1);
        bool numeric = lua_isnumber(L, -1) != 0;
        lua_pop(L, 1);
        
        if (numeric) {
            lua_rawgeti(L, tblIdx, 1); v.x = (float)lua_tonumber(L, -1); lua_pop(L, 1);
            lua_rawgeti(L, tblIdx, 2); v.y = (float)lua_tonumber(L, -1); lua_pop(L, 1);
            lua_rawgeti(L, tblIdx, 3); v.z = (float)lua_tonumber(L, -1); lua_pop(L, 1);
            lua_rawgeti(L, tblIdx, 4); v.r = (unsigned char)lua_tointeger(L, -1); lua_pop(L, 1);
            lua_rawgeti(L, tblIdx, 5); v.g = (unsigned char)lua_tointeger(L, -1); lua_pop(L, 1);
            lua_rawgeti(L, tblIdx, 6); v.b = (unsigned char)lua_tointeger(L, -1); lua_pop(L, 1);
            lua_rawgeti(L, tblIdx, 7); v.a = (unsigned char)lua_tointeger(L, -1); lua_pop(L, 1);
        } else {
            v.x = (float)getNum(L, tblIdx, "x", 0);
            v.y = (float)getNum(L, tblIdx, "y", 0);
            v.z = (float)getNum(L, tblIdx, "z", 0);
            lua_getfield(L, tblIdx, "color");
            if (lua_istable(L, -1)) {
                v.r = (unsigned char)getFieldInteger(L, -1, "r", 255);
                v.g = (unsigned char)getFieldInteger(L, -1, "g", 255);
                v.b = (unsigned char)getFieldInteger(L, -1, "b", 255);
                v.a = (unsigned char)getFieldInteger(L, -1, "a", 255);
            }
            lua_pop(L, 1);
        }
        
        if (v.x < minX) minX = v.x; if (v.x > maxX) maxX = v.x;
        if (v.y < minY) minY = v.y; if (v.y > maxY) maxY = v.y;
        if (v.z < minZ) minZ = v.z; if (v.z > maxZ) maxZ = v.z;
        
        voxels.push_back(v);
        lua_pop(L, 1);
    }
    
    // Build model context
    modelCtx.minX = (int)minX; modelCtx.minY = (int)minY; modelCtx.minZ = (int)minZ;
    modelCtx.sizeX = (int)(maxX - minX + 1);
    modelCtx.sizeY = (int)(maxY - minY + 1);
    modelCtx.sizeZ = (int)(maxZ - minZ + 1);
    for (auto& v : voxels) {
        modelCtx.voxel_map[pack_coords((int)v.x, (int)v.y, (int)v.z)] = &v;
    }
    
    // Build position→voxel-index map for edit mode ID map
    std::unordered_map<uint64_t, uint32_t> posToVoxelIdx;
    if (editMode) {
        for (size_t i = 0; i < voxels.size(); i++) {
            posToVoxelIdx[pack_coords((int)voxels[i].x, (int)voxels[i].y, (int)voxels[i].z)] = (uint32_t)(i + 1);
        }
    }
    
    // Create result table
    lua_createtable(L, 0, 3);
    
    if (count == 0) {
        for (auto& s : active_shaders) {
            if (s.iface && s.instance) s.iface->destroy(s.instance);
        }
        lua_pushinteger(L, width); lua_setfield(L, -2, "width");
        lua_pushinteger(L, height); lua_setfield(L, -2, "height");
        lua_pushlstring(L, "", 0); lua_setfield(L, -2, "pixels");
        return 1;
    }
    
    // Calculate model center and camera setup
    // Read custom middlePoint from params if provided, otherwise calculate from bounds
    float midX, midY, midZ;
    lua_getfield(L, 2, "middlePoint");
    if (lua_istable(L, -1)) {
        midX = (float)getNum(L, -1, "x", 0.5f * (minX + maxX));
        midY = (float)getNum(L, -1, "y", 0.5f * (minY + maxY));
        midZ = (float)getNum(L, -1, "z", 0.5f * (minZ + maxZ));
    } else {
        midX = 0.5f * (minX + maxX);
        midY = 0.5f * (minY + maxY);
        midZ = 0.5f * (minZ + maxZ);
    }
    lua_pop(L, 1);
    // Use globalBounds (if provided) for camera calculations
    CameraBounds camB = computeCameraBounds(L, 2, minX, maxX, minY, maxY, minZ, maxZ, midX, midY, midZ);
    float sizeX = camB.sizeX;
    float sizeY = camB.sizeY;
    float sizeZ = camB.sizeZ;
    float maxDim = camB.maxDim;
    float effectiveSpan = camB.effectiveSpan;
    
    float RX = xRotDeg * (float)PI / 180.f;
    float RY = yRotDeg * (float)PI / 180.f;
    float RZ = zRotDeg * (float)PI / 180.f;
    float cx = std::cos(RX), sx = std::sin(RX);
    float cy = std::cos(RY), sy = std::sin(RY);
    float cz = std::cos(RZ), sz = std::sin(RZ);
    
    bool perspective = (!orth && fovDeg > 0.f);
    float focalLength = 0.f;
    float camDist;
    
    if (perspective) {
        fovDeg = std::max(5.f, std::min(75.f, fovDeg));
        float warpT = (fovDeg - 5.f) / (75.f - 5.f);
        float amplified = std::pow(warpT, 1.f / 3.f);
        const float BASE_NEAR = 1.2f;
        const float FAR_EXTRA = 45.f;
        camDist = effectiveSpan * (BASE_NEAR + (1.f - amplified) * (1.f - amplified) * FAR_EXTRA);
        camDist *= perspIntensity;  // Apply perspective intensity multiplier
        float fovRad = fovDeg * (float)PI / 180.f;
        focalLength = (height / 2.f) / std::tan(fovRad / 2.f);
    } else {
        camDist = effectiveSpan * 5.f;
    }
    
    float voxelSize = scale;
    {
        float maxAllowed = std::min(width, height) * viewportCutoffMultiplier;
        if (perspective && focalLength > 1e-6f && maxDim > 0.f) {
            // Rotate AABB corners to derive front/back depths (match l_render_basic)
            float zMinRot =  1e9f;
            float zMaxRot = -1e9f;
            for (int ix = 0; ix < 2; ++ix) {
                float cxv = (ix == 0) ? camB.camMinX : camB.camMaxX;
                for (int iy = 0; iy < 2; ++iy) {
                    float cyv = (iy == 0) ? camB.camMinY : camB.camMaxY;
                    for (int iz = 0; iz < 2; ++iz) {
                        float czv = (iz == 0) ? camB.camMinZ : camB.camMaxZ;
                        float X = cxv - midX;
                        float Y = cyv - midY;
                        float Z = czv - midZ;
                        { float Y2 = Y*cx - Z*sx; float Z2 = Y*sx + Z*cx; Y = Y2; Z = Z2; }
                        { float X2 = X*cy + Z*sy; float Z3 = -X*sy + Z*cy; X = X2; Z = Z3; }
                        { float X3 = X*cz - Y*sz; float Y3 = X*sz + Y*cz; X = X3; Y = Y3; }
                        float worldZ = Z + midZ;
                        if (worldZ < zMinRot) zMinRot = worldZ;
                        if (worldZ > zMaxRot) zMaxRot = worldZ;
                    }
                }
            }
            float cameraZ = midZ + camDist;
            float depthBack   = std::max(0.001f, cameraZ - zMinRot);  // farthest
            float depthFront  = std::max(0.001f, cameraZ - zMaxRot);  // nearest
            float depthMiddle = std::max(0.001f, camDist);
            float depthRef = depthMiddle;
            if (perspRef == "front" || perspRef == "Front") depthRef = depthFront;
            else if (perspRef == "back" || perspRef == "Back") depthRef = depthBack;
            voxelSize = scale * (depthRef / focalLength);
            if (voxelSize * effectiveSpan > maxAllowed) voxelSize = maxAllowed / effectiveSpan;
            if (voxelSize <= 0.f) voxelSize = 1.f;
        } else {
            if (voxelSize < 1) voxelSize = 1;
            if (voxelSize * effectiveSpan > maxAllowed) voxelSize *= (maxAllowed / (voxelSize * effectiveSpan));
        }
    }
    // Use slightly negative threshold to ensure edge-on faces are consistently visible
    // This prevents jagged edges at certain rotation angles where faces are nearly perpendicular
    float threshold = -0.001f;
    
    // ========================================================================
    // Setup native context for shaders
    // ========================================================================
    native_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.width = width;
    ctx.height = height;
    ctx.model = &modelCtx;
    ctx.num_lights = 0;
    ctx.q_view[0] = 0; ctx.q_view[1] = 0; ctx.q_view[2] = 0; ctx.q_view[3] = 1;
    
    // Rotation components for view-to-model transforms (used by occlusion)
    // Array: [cos(x), sin(x), cos(y), sin(y), cos(z), sin(z)]
    double rotation_components[6] = {
        (double)cx, (double)sx,
        (double)cy, (double)sy,
        (double)cz, (double)sz
    };
    ctx.rotation = rotation_components;
    
    // Set model geometry info for shaders (radial attenuation etc.)
    ctx.model_center[0] = midX;
    ctx.model_center[1] = midY;
    ctx.model_center[2] = midZ;
    float diag = std::sqrt(sizeX*sizeX + sizeY*sizeY + sizeZ*sizeZ);
    ctx.model_radius = 0.5f * diag;
    
    // Parse lighting from params
    lua_getfield(L, 2, "lighting");
    if (lua_istable(L, -1)) {
        float pitch = (float)getNum(L, -1, "pitch", 0.0);
        float yaw = (float)getNum(L, -1, "yaw", 0.0);
        float yawRad = yaw * (float)PI / 180.f;
        float pitchRad = pitch * (float)PI / 180.f;
        float cosYaw = std::cos(yawRad), sinYaw = std::sin(yawRad);
        float cosPitch = std::cos(pitchRad), sinPitch = std::sin(pitchRad);
        
        ctx.num_lights = 1;
        ctx.lights[0].dir[0] = cosYaw * cosPitch;
        ctx.lights[0].dir[1] = sinPitch;
        ctx.lights[0].dir[2] = sinYaw * cosPitch;
        ctx.lights[0].intensity = 1.0f;
        ctx.lights[0].spec_power = 1.0f;
    }
    lua_pop(L, 1);
    
    // Run pre-hooks
    for (auto& s : active_shaders) {
        if (s.iface->run_pre) s.iface->run_pre(s.instance, &ctx);
    }
    
    // ========================================================================
    // Phase 3: Build face list from visibility data or compute directly
    // ========================================================================
    std::vector<native_face_data_t> batch_faces;
    batch_faces.reserve(voxels.size() * 6);
    
    // Try to parse pre-computed visibility data
    parseVisibilityData(L, 2, batch_faces, &modelCtx);
    
    float screenCX = width * 0.5f;
    float screenCY = height * 0.5f;
    float camZ = midZ + camDist;
    
    // If visibility data was provided, post-process faces to transform normals and compute depth
    if (!batch_faces.empty()) {
        std::vector<native_face_data_t> culled_faces;
        culled_faces.reserve(batch_faces.size());
        
        for (auto& face : batch_faces) {
            // Transform voxel position
            float x = face.x - midX, y = face.y - midY, z = face.z - midZ;
            { float y2 = y * cx - z * sx; float z2 = y * sx + z * cx; y = y2; z = z2; }
            { float x2 = x * cy + z * sy; float z3 = -x * sy + z * cy; x = x2; z = z3; }
            { float x3 = x * cz - y * sz; float y3 = x * sz + y * cz; x = x3; y = y3; }
            float worldX = x + midX, worldY = y + midY, worldZ = z + midZ;
            
            // Transform normal to world space
            float nx = face.normal[0], ny = face.normal[1], nz = face.normal[2];
            { float y2 = ny * cx - nz * sx; float z2 = ny * sx + nz * cx; ny = y2; nz = z2; }
            { float x2 = nx * cy + nz * sy; float z3 = -nx * sy + nz * cy; nx = x2; nz = z3; }
            { float x3 = nx * cz - ny * sz; float y3 = nx * sz + ny * cz; nx = x3; ny = y3; }
            
            // Backface culling
            float vxv, vyv, vzv;
            if (orth) { vxv = 0.f; vyv = 0.f; vzv = 1.f; }
            else {
                vxv = midX - worldX; vyv = midY - worldY; vzv = camZ - worldZ;
                float m = std::sqrt(vxv * vxv + vyv * vyv + vzv * vzv);
                if (m > 1e-5f) { vxv /= m; vyv /= m; vzv /= m; }
            }
            float dot = nx * vxv + ny * vyv + nz * vzv;
            if (dot <= threshold) continue;  // Backface culled
            
            // Store rotated normal
            face.normal[0] = nx; face.normal[1] = ny; face.normal[2] = nz;
            
            // Compute depth
            face.depth = orth ? (camZ - worldZ) : std::max(0.001f, camZ - worldZ);
            
            culled_faces.push_back(face);
        }
        batch_faces = std::move(culled_faces);
    }
    
    // If no visibility data provided, compute faces directly (like original render_stack)
    if (batch_faces.empty()) {
        // Neighbor offsets for interior face culling - matches face order in LOCAL_FACE_NORMALS
        // face 0 = front (+Z), face 1 = back (-Z), face 2 = right (+X), face 3 = left (-X), face 4 = top (+Y), face 5 = bottom (-Y)
        static const int NEIGHBOR_OFFSET[6][3] = {
            {0, 0, 1},   // front: neighbor at +Z
            {0, 0, -1},  // back: neighbor at -Z
            {1, 0, 0},   // right: neighbor at +X
            {-1, 0, 0},  // left: neighbor at -X
            {0, 1, 0},   // top: neighbor at +Y
            {0, -1, 0}   // bottom: neighbor at -Y
        };
        
        for (auto& v : voxels) {
            int vx = (int)v.x, vy = (int)v.y, vz = (int)v.z;
            
            float x = v.x - midX, y = v.y - midY, z = v.z - midZ;
            { float y2 = y * cx - z * sx; float z2 = y * sx + z * cx; y = y2; z = z2; }
            { float x2 = x * cy + z * sy; float z3 = -x * sy + z * cy; x = x2; z = z3; }
            { float x3 = x * cz - y * sz; float y3 = x * sz + y * cz; x = x3; y = y3; }
            float worldX = x + midX, worldY = y + midY, worldZ = z + midZ;
            
            float vxv, vyv, vzv;
            if (orth) { vxv = 0.f; vyv = 0.f; vzv = 1.f; }
            else {
                vxv = midX - worldX; vyv = midY - worldY; vzv = camZ - worldZ;
                float m = std::sqrt(vxv * vxv + vyv * vyv + vzv * vzv);
                if (m > 1e-5f) { vxv /= m; vyv /= m; vzv /= m; }
            }
            
            for (int f = 0; f < 6; f++) {
                // Interior face culling: skip face if neighbor voxel exists in that direction
                int nx_off = vx + NEIGHBOR_OFFSET[f][0];
                int ny_off = vy + NEIGHBOR_OFFSET[f][1];
                int nz_off = vz + NEIGHBOR_OFFSET[f][2];
                if (modelCtx.voxel_map.count(pack_coords(nx_off, ny_off, nz_off))) {
                    continue;  // Neighbor exists, face is occluded
                }
                
                float nx = LOCAL_FACE_NORMALS[f][0];
                float ny = LOCAL_FACE_NORMALS[f][1];
                float nz = LOCAL_FACE_NORMALS[f][2];
                {
                    float y2 = ny * cx - nz * sx; float z2 = ny * sx + nz * cx; ny = y2; nz = z2;
                    float x2 = nx * cy + nz * sy; float z3 = -nx * sy + nz * cy; nx = x2; nz = z3;
                    float x3 = nx * cz - ny * sz; float y3 = nx * sz + ny * cz; nx = x3; ny = y3;
                }
                float dot = nx * vxv + ny * vyv + nz * vzv;
                if (dot <= threshold) continue;  // Backface culling
                
                native_face_data_t face;
                face.x = vx; face.y = vy; face.z = vz;
                
                face.face_dir = internalFaceToApiDir(f);
                
                face.r = v.r; face.g = v.g; face.b = v.b; face.a = v.a;
                face.normal[0] = nx; face.normal[1] = ny; face.normal[2] = nz;
                
                // Compute face depth as average of all 4 rotated vertex Z coords
                float avgDepth = 0.f;
                for (int vi = 0; vi < 4; vi++) {
                    int vid = FACE_IDX[f][vi] - 1;
                    float lx = UNIT_VERTS[vid][0] * voxelSize;
                    float ly = UNIT_VERTS[vid][1] * voxelSize;
                    float lz = UNIT_VERTS[vid][2] * voxelSize;
                    // Apply full rotation to vertex offset
                    { float y2 = ly * cx - lz * sx; float z2 = ly * sx + lz * cx; ly = y2; lz = z2; }
                    { float x2 = lx * cy + lz * sy; float z3 = -lx * sy + lz * cy; lx = x2; lz = z3; }
                    { float x3 = lx * cz - ly * sz; float y3 = lx * sz + ly * cz; lx = x3; ly = y3; }
                    float wz = worldZ + (lz / voxelSize);
                    avgDepth += (camZ - wz);
                }
                face.depth = perspective ? std::max(0.001f, avgDepth / 4.f) : avgDepth / 4.f;
                
                batch_faces.push_back(face);
            }
        }
    }
    
    // ========================================================================
    // Execute shader pipeline WITH COLOR ROUTING
    // Supports colorInput/outputName/addTo for shader chaining
    // ========================================================================
    if (!batch_faces.empty()) {
        // Color buffer type: stores RGBA per face
        struct ColorBuffer {
            std::vector<unsigned char> r, g, b, a;
            void resize(size_t n) { r.resize(n); g.resize(n); b.resize(n); a.resize(n); }
            void copyFrom(const std::vector<native_face_data_t>& faces) {
                resize(faces.size());
                for (size_t i = 0; i < faces.size(); i++) {
                    r[i] = faces[i].r; g[i] = faces[i].g; 
                    b[i] = faces[i].b; a[i] = faces[i].a;
                }
            }
            void applyTo(std::vector<native_face_data_t>& faces) const {
                for (size_t i = 0; i < faces.size() && i < r.size(); i++) {
                    faces[i].r = r[i]; faces[i].g = g[i];
                    faces[i].b = b[i]; faces[i].a = a[i];
                }
            }
        };
        
        // Named outputs dictionary
        std::unordered_map<std::string, ColorBuffer> outputs;
        
        // Store base colors (original voxel colors)
        ColorBuffer baseColors;
        baseColors.copyFrom(batch_faces);
        outputs["baseColor"] = baseColors;
        
        // Track "last" output
        ColorBuffer lastColors = baseColors;
        
        // Helper to get effective output name
        auto getEffectiveName = [](const ActiveShader& s) -> std::string {
            if (!s.outputName.empty()) return s.outputName;
            // Auto-generate: <ShaderName> #N
            std::string name = s.id;
            size_t dot = name.rfind('.');
            if (dot != std::string::npos) name = name.substr(dot + 1);
            if (!name.empty()) name[0] = toupper(name[0]);
            return "<" + name + "> #" + std::to_string(s.localIndex + 1);
        };
        
        // Helper to get source colors
        auto getSourceColors = [&](const std::string& sourceName) -> const ColorBuffer& {
            if (sourceName == "last" || sourceName.empty()) return lastColors;
            if (sourceName == "baseColor") return baseColors;
            auto it = outputs.find(sourceName);
            if (it != outputs.end()) return it->second;
            return lastColors; // fallback
        };
        
        // Helper for additive combination
        auto combineAdditive = [](const ColorBuffer& addend, const ColorBuffer& target) -> ColorBuffer {
            ColorBuffer result;
            size_t n = addend.r.size();
            result.resize(n);
            for (size_t i = 0; i < n; i++) {
                result.r[i] = (unsigned char)std::min(255, (int)addend.r[i] + (int)target.r[i]);
                result.g[i] = (unsigned char)std::min(255, (int)addend.g[i] + (int)target.g[i]);
                result.b[i] = (unsigned char)std::min(255, (int)addend.b[i] + (int)target.b[i]);
                result.a[i] = addend.a[i];
            }
            return result;
        };
        
        // ====================================================================
        // TOPOLOGICAL SORT with Kahn's algorithm
        // Dependencies determined by colorInput references
        // Original order (category + index) used as tiebreaker only
        // ====================================================================
        
        // Build name -> shader index map
        std::unordered_map<std::string, size_t> nameToIdx;
        std::vector<std::string> names;
        for (size_t i = 0; i < active_shaders.size(); i++) {
            std::string name = getEffectiveName(active_shaders[i]);
            nameToIdx[name] = i;
            names.push_back(name);
        }
        
        // Build adjacency list and in-degree count
        // Edge: dependency -> dependent (if B depends on A, edge A->B)
        std::vector<std::vector<size_t>> adjList(active_shaders.size());
        std::vector<int> inDegree(active_shaders.size(), 0);
        
        for (size_t i = 0; i < active_shaders.size(); i++) {
            const std::string& colorInput = active_shaders[i].colorInput;
            // Check if colorInput references another shader's output
            if (!colorInput.empty() && colorInput != "baseColor" && colorInput != "last") {
                auto it = nameToIdx.find(colorInput);
                if (it != nameToIdx.end()) {
                    // Shader i depends on shader it->second
                    adjList[it->second].push_back(i);
                    inDegree[i]++;
                }
            }
        }
        
        // Kahn's algorithm with priority queue for tiebreaking
        // Priority: lower original index = higher priority (runs first among same-level)
        // But we want: lighting before fx at same level, then by localIndex
        auto getPriority = [&](size_t idx) -> int {
            // Lower priority value = runs first
            // fx=0, lighting=1000 base, then add localIndex
            // Wait, we want dependencies to determine order, category only as tiebreaker
            // Among shaders with no dependencies between them, use original order
            const auto& s = active_shaders[idx];
            int catPriority = (s.category == "lighting") ? 0 : 1000;
            return catPriority + s.localIndex;
        };
        
        // Use a set sorted by priority for the "ready" queue
        auto cmp = [&](size_t a, size_t b) {
            return getPriority(a) < getPriority(b);
        };
        std::vector<size_t> ready;
        for (size_t i = 0; i < active_shaders.size(); i++) {
            if (inDegree[i] == 0) ready.push_back(i);
        }
        std::sort(ready.begin(), ready.end(), cmp);
        
        std::vector<ActiveShader*> execOrder;
        while (!ready.empty()) {
            // Take the shader with highest priority (lowest priority value)
            size_t curr = ready.front();
            ready.erase(ready.begin());
            execOrder.push_back(&active_shaders[curr]);
            
            // Process all dependents
            for (size_t dep : adjList[curr]) {
                inDegree[dep]--;
                if (inDegree[dep] == 0) {
                    ready.push_back(dep);
                    std::sort(ready.begin(), ready.end(), cmp);
                }
            }
        }
        
        // Execute each shader with color routing
        for (auto* sp : execOrder) {
            ActiveShader& s = *sp;
            if (!s.iface->run_voxel_batch) continue;
            
            std::string effName = getEffectiveName(s);
            printf("[native] Executing shader: %s (colorInput=%s)\n", effName.c_str(), s.colorInput.c_str());
            
            // Get input colors based on colorInput
            const ColorBuffer& inputColors = getSourceColors(s.colorInput);
            
            // Debug: print first face color before apply
            if (!batch_faces.empty()) {
                printf("[native]   Before applyTo: face[0] = (%d,%d,%d)\n", 
                       batch_faces[0].r, batch_faces[0].g, batch_faces[0].b);
                printf("[native]   inputColors[0] = (%d,%d,%d)\n",
                       inputColors.r[0], inputColors.g[0], inputColors.b[0]);
            }
            
            // Apply input colors to faces
            inputColors.applyTo(batch_faces);
            
            // Debug: print first face color after apply
            if (!batch_faces.empty()) {
                printf("[native]   After applyTo: face[0] = (%d,%d,%d)\n", 
                       batch_faces[0].r, batch_faces[0].g, batch_faces[0].b);
            }
            
            // Execute shader (modifies batch_faces in place)
            s.iface->run_voxel_batch(s.instance, &ctx, (int)batch_faces.size(), batch_faces.data());
            
            // Debug: print first face color after shader
            if (!batch_faces.empty()) {
                printf("[native]   After shader run: face[0] = (%d,%d,%d)\n", 
                       batch_faces[0].r, batch_faces[0].g, batch_faces[0].b);
            }
            
            // Capture output colors
            ColorBuffer outputColors;
            outputColors.copyFrom(batch_faces);
            
            // Debug: print what we're storing
            printf("[native]   Storing output as: %s\n", getEffectiveName(s).c_str());
            
            // Handle addTo: additive accumulation
            if (!s.addTo.empty()) {
                auto it = outputs.find(s.addTo);
                if (it != outputs.end()) {
                    outputColors = combineAdditive(outputColors, it->second);
                    outputColors.applyTo(batch_faces);
                }
            }
            
            // Store output with effective name
            std::string effectiveName = getEffectiveName(s);
            outputs[effectiveName] = outputColors;
            
            // Update "last"
            lastColors = outputColors;
        }
    }
    
    // ========================================================================
    // Convert faces to polygons for rasterization
    // ========================================================================
    struct Poly { 
        float x[4], y[4], depth; 
        unsigned char r, g, b, a;
        // Tiebreaker fields for stable sorting
        int voxelX, voxelY, voxelZ;
        int faceIdx;
        uint32_t voxelId; // Edit-mode: 1-based voxel array index
    };
    std::vector<Poly> polys;
    polys.reserve(batch_faces.size());
    
    for (const auto& face : batch_faces) {
        Poly poly;
        poly.r = face.r; poly.g = face.g; poly.b = face.b; poly.a = face.a;
        poly.depth = face.depth;
        poly.voxelX = face.x;
        poly.voxelY = face.y;
        poly.voxelZ = face.z;
        poly.voxelId = 0;
        if (editMode) {
            auto it = posToVoxelIdx.find(pack_coords(face.x, face.y, face.z));
            if (it != posToVoxelIdx.end()) poly.voxelId = it->second;
        }
        
        // Map API face_dir back to internal face index
        // Internal: 0=front(+Z), 1=back(-Z), 2=right(+X), 3=left(-X), 4=top(+Y), 5=bottom(-Y)
        int f = 0;
        switch (face.face_dir) {
            case 4: f = 0; break; // +Z -> front
            case 5: f = 1; break; // -Z -> back
            case 0: f = 2; break; // +X -> right
            case 1: f = 3; break; // -X -> left
            case 2: f = 4; break; // +Y -> top
            case 3: f = 5; break; // -Y -> bottom
        }
        poly.faceIdx = f; // Use internal face index (consistent with basic/fast paths)
        
        float x = face.x - midX;
        float y = face.y - midY;
        float z = face.z - midZ;
        { float y2 = y * cx - z * sx; float z2 = y * sx + z * cx; y = y2; z = z2; }
        { float x2 = x * cy + z * sy; float z3 = -x * sy + z * cy; x = x2; z = z3; }
        { float x3 = x * cz - y * sz; float y3 = x * sz + y * cz; x = x3; y = y3; }
        float worldX = x + midX, worldY = y + midY, worldZ = z + midZ;
        
        for (int vi = 0; vi < 4; vi++) {
            int vid = FACE_IDX[f][vi] - 1;
            float lx = UNIT_VERTS[vid][0] * voxelSize;
            float ly = UNIT_VERTS[vid][1] * voxelSize;
            float lz = UNIT_VERTS[vid][2] * voxelSize;
            { float y2 = ly * cx - lz * sx; float z2 = ly * sx + lz * cx; ly = y2; lz = z2; }
            { float x2 = lx * cy + lz * sy; float z3 = -lx * sy + lz * cy; lx = x2; lz = z3; }
            { float x3 = lx * cz - ly * sz; float y3 = lx * sz + ly * cz; lx = x3; ly = y3; }
            float wx = worldX * voxelSize + lx;
            float wy = worldY * voxelSize + ly;
            float wz = worldZ + (lz / voxelSize);
            if (perspective) {
                float depth = std::max(0.001f, camZ - wz);
                float s = focalLength > 0 ? (focalLength / depth) : 1.f;
                poly.x[vi] = screenCX + (wx - midX * voxelSize) * s;
                poly.y[vi] = screenCY + (wy - midY * voxelSize) * s;
            } else {
                poly.x[vi] = screenCX + (wx - midX * voxelSize);
                poly.y[vi] = screenCY + (wy - midY * voxelSize);
            }
        }
        polys.push_back(poly);
    }
    
    // --- Preview voxels (ghost style, skip shader pipeline) ---
    {
        auto previewVoxels = readPreviewVoxels(L, 2);
        if (!previewVoxels.empty()) {
            // Build combined occupancy for preview interior culling
            std::unordered_set<long long> combinedOcc;
            for (auto& v : voxels) combinedOcc.insert(pack_coords((int)v.x, (int)v.y, (int)v.z));
            for (auto& pv : previewVoxels) combinedOcc.insert(pack_coords((int)pv.x, (int)pv.y, (int)pv.z));

            for (auto& pv : previewVoxels) {
                int vx = (int)pv.x, vy = (int)pv.y, vz = (int)pv.z;
                float x = pv.x - midX, y = pv.y - midY, z = pv.z - midZ;
                { float y2 = y * cx - z * sx; float z2 = y * sx + z * cx; y = y2; z = z2; }
                { float x2 = x * cy + z * sy; float z3 = -x * sy + z * cy; x = x2; z = z3; }
                { float x3 = x * cz - y * sz; float y3 = x * sz + y * cz; x = x3; y = y3; }
                float worldX = x + midX, worldY = y + midY, worldZ = z + midZ;

                float vxv, vyv, vzv;
                if (orth) { vxv = 0.f; vyv = 0.f; vzv = 1.f; }
                else {
                    vxv = midX - worldX; vyv = midY - worldY; vzv = camZ - worldZ;
                    float m = std::sqrt(vxv * vxv + vyv * vyv + vzv * vzv);
                    if (m > 1e-5f) { vxv /= m; vyv /= m; vzv /= m; }
                }

                for (int f = 0; f < 6; f++) {
                    int nx_off = vx + PREVIEW_NEIGH[f][0];
                    int ny_off = vy + PREVIEW_NEIGH[f][1];
                    int nz_off = vz + PREVIEW_NEIGH[f][2];
                    if (combinedOcc.count(pack_coords(nx_off, ny_off, nz_off))) continue;

                    float fnx = LOCAL_FACE_NORMALS[f][0], fny = LOCAL_FACE_NORMALS[f][1], fnz = LOCAL_FACE_NORMALS[f][2];
                    { float y2 = fny * cx - fnz * sx; float z2 = fny * sx + fnz * cx; fny = y2; fnz = z2; }
                    { float x2 = fnx * cy + fnz * sy; float z3 = -fnx * sy + fnz * cy; fnx = x2; fnz = z3; }
                    { float x3 = fnx * cz - fny * sz; float y3 = fnx * sz + fny * cz; fnx = x3; fny = y3; }
                    float dot = fnx * vxv + fny * vyv + fnz * vzv;
                    if (dot <= threshold) continue;

                    Poly poly;
                    poly.r = pv.r; poly.g = pv.g; poly.b = pv.b; poly.a = pv.a;
                    poly.voxelX = vx; poly.voxelY = vy; poly.voxelZ = vz;
                    poly.faceIdx = f;
                    poly.voxelId = 0;

                    for (int vi = 0; vi < 4; vi++) {
                        int vid = FACE_IDX[f][vi] - 1;
                        float lx = UNIT_VERTS[vid][0] * voxelSize;
                        float ly = UNIT_VERTS[vid][1] * voxelSize;
                        float lz = UNIT_VERTS[vid][2] * voxelSize;
                        { float y2 = ly * cx - lz * sx; float z2 = ly * sx + lz * cx; ly = y2; lz = z2; }
                        { float x2 = lx * cy + lz * sy; float z3 = -lx * sy + lz * cy; lx = x2; lz = z3; }
                        { float x3 = lx * cz - ly * sz; float y3 = lx * sz + ly * cz; lx = x3; ly = y3; }
                        float wx = worldX * voxelSize + lx;
                        float wy = worldY * voxelSize + ly;
                        float wz = worldZ + (lz / voxelSize);
                        if (perspective) {
                            float depth = std::max(0.001f, camZ - wz);
                            float s = focalLength > 0 ? (focalLength / depth) : 1.f;
                            poly.x[vi] = screenCX + (wx - midX * voxelSize) * s;
                            poly.y[vi] = screenCY + (wy - midY * voxelSize) * s;
                        } else {
                            poly.x[vi] = screenCX + (wx - midX * voxelSize);
                            poly.y[vi] = screenCY + (wy - midY * voxelSize);
                        }
                    }
                    poly.depth = orth ? (camZ - worldZ) : std::max(0.001f, camZ - worldZ);
                    polys.push_back(poly);
                }
            }
        }
    }
    
    // Run post-hooks
    for (auto& s : active_shaders) {
        if (s.iface->run_post) s.iface->run_post(s.instance, &ctx);
    }
    
    // Cleanup shader instances
    for (auto& s : active_shaders) {
        if (s.iface && s.instance) s.iface->destroy(s.instance);
    }
    
    // Stable sort: primary by depth (far to near), secondary by voxel position + face for tiebreaker
    std::sort(polys.begin(), polys.end(), [](const Poly& a, const Poly& b) {
        constexpr float DEPTH_EPS = 0.0001f;
        if (std::abs(a.depth - b.depth) > DEPTH_EPS) {
            return a.depth > b.depth;
        }
        // Tiebreaker: consistent ordering by voxel position, then face index
        if (a.voxelZ != b.voxelZ) return a.voxelZ < b.voxelZ;
        if (a.voxelY != b.voxelY) return a.voxelY < b.voxelY;
        if (a.voxelX != b.voxelX) return a.voxelX < b.voxelX;
        return a.faceIdx < b.faceIdx;
    });
    
    std::vector<unsigned char> buffer((size_t)width * height * 4, 0);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            size_t off = ((size_t)y * width + x) * 4;
            buffer[off + 0] = bgR; buffer[off + 1] = bgG; buffer[off + 2] = bgB; buffer[off + 3] = bgA;
        }
    }
    
    // Allocate ID map buffer for edit mode
    std::vector<uint32_t> idMapBuf;
    std::vector<uint32_t>* idMapPtr = nullptr;
    if (editMode) {
        idMapBuf.resize((size_t)width * height, 0);
        idMapPtr = &idMapBuf;
    }
    
    for (auto& p : polys) {
        FacePoly polyOut;
        for (int i = 0; i < 4; i++) { polyOut.x[i] = p.x[i]; polyOut.y[i] = p.y[i]; }
        polyOut.depth = p.depth; polyOut.r = p.r; polyOut.g = p.g; polyOut.b = p.b; polyOut.a = p.a;
        polyOut.voxelId = p.voxelId;
        polyOut.faceIdx = p.faceIdx;
        rasterQuad(polyOut, width, height, buffer, idMapPtr);
    }
    
    lua_pushinteger(L, width); lua_setfield(L, -2, "width");
    lua_pushinteger(L, height); lua_setfield(L, -2, "height");
    lua_pushlstring(L, (const char*)buffer.data(), buffer.size()); lua_setfield(L, -2, "pixels");
    if (editMode) {
        lua_pushlstring(L, (const char*)idMapBuf.data(), idMapBuf.size() * sizeof(uint32_t));
        lua_setfield(L, -2, "editIdMap");
    }
    return 1;
}

// ============================================================================
// Debug logging control
// ============================================================================
static int l_set_debug_shader_params(lua_State* L) {
    g_debug_shader_params = lua_toboolean(L, 1) != 0;
    printf("[native] Shader param debug: %s\n", g_debug_shader_params ? "ON" : "OFF");
    lua_pushboolean(L, g_debug_shader_params);
    return 1;
}

// Alias for set_debug_shader_params with consistent naming
static int l_set_debug_logging(lua_State* L) {
    return l_set_debug_shader_params(L);
}

static int l_get_shader_info(lua_State* L) {
    // Return info about loaded native shaders
    lua_newtable(L);
    
    int count = native_shader_loader::get_shader_count();
    for (int i = 0; i < count; i++) {
        const char* shader_id = native_shader_loader::get_shader_id(i);
        if (!shader_id) continue;
        
        const native_shader_v1_t* iface = native_shader_loader::get_shader_interface(shader_id);
        
        lua_newtable(L);
        lua_pushstring(L, shader_id);
        lua_setfield(L, -2, "id");
        
        if (iface) {
            if (iface->shader_id) {
                lua_pushstring(L, iface->shader_id());
                lua_setfield(L, -2, "native_id");
            }
            if (iface->display_name) {
                lua_pushstring(L, iface->display_name());
                lua_setfield(L, -2, "name");
            }
            
            // Get parameter schema
            if (iface->params_schema) {
                int param_count = 0;
                const native_param_def_t* params = iface->params_schema(&param_count);
                if (params && param_count > 0) {
                    lua_newtable(L);
                    for (int p = 0; p < param_count; p++) {
                        lua_newtable(L);
                        lua_pushstring(L, params[p].key);
                        lua_setfield(L, -2, "key");
                        lua_pushinteger(L, params[p].type);
                        lua_setfield(L, -2, "type");
                        if (params[p].display_name) {
                            lua_pushstring(L, params[p].display_name);
                            lua_setfield(L, -2, "display_name");
                        }
                        lua_rawseti(L, -2, p + 1);
                    }
                    lua_setfield(L, -2, "params");
                }
            }
        }
        
        lua_rawseti(L, -2, i + 1);
    }
    
    // Also push the count
    lua_pushinteger(L, count);
    lua_setfield(L, -2, "count");
    
    return 1;
}

// ============================================================================
// FAST RENDERER: High-performance voxel rendering using axis-aligned rectangles
// Instead of per-face polygon rasterization, each voxel is rendered as a single
// filled rectangle (1 draw call per voxel vs 6 faces × polygon rasterization).
//
// SHADER BEHAVIOR:
//   - hasShaders=false: use raw voxel colors (NO shading applied)
//   - hasShaders=true: execute full shader stack (lighting + fx) on voxel batch
// ============================================================================
static int l_render_fast(lua_State* L) {
    if (!lua_istable(L, 1) || !lua_istable(L, 2)) {
        lua_pushnil(L); lua_pushstring(L, "expected (voxels, params)"); return 2;
    }
    
    int width = (int)getNum(L, 2, "width", 200);
    int height = (int)getNum(L, 2, "height", 200);
    float scale = (float)getNum(L, 2, "scale", -12345.0);
    if (scale < 0) scale = (float)getNum(L, 2, "scaleLevel", 1.0);
    if (scale <= 0.f) scale = 1.f;
    
    float xRotDeg = (float)getNum(L, 2, "xRotation", 0.0);
    float yRotDeg = (float)getNum(L, 2, "yRotation", 0.0);
    float zRotDeg = (float)getNum(L, 2, "zRotation", 0.0);
    float shadeIntensity = (float)getNum(L, 2, "basicShadeIntensity", 100.0);
    float lightIntensity = (float)getNum(L, 2, "basicLightIntensity", 0.0);
    float voxelPadding = (float)getNum(L, 2, "voxelPadding", 2.0); // Default 2 to match aseSlab
    
    // Parse editMode for ID map output
    bool editMode = false;
    lua_getfield(L, 2, "editMode");
    if (lua_isboolean(L, -1)) editMode = lua_toboolean(L, -1) != 0;
    lua_pop(L, 1);
    
    // Check if shader stack has any enabled shaders
    // If hasShaders is false, we render flat colors with no shading
    bool hasShaders = false;
    lua_getfield(L, 2, "hasShaders");
    if (!lua_isnil(L, -1)) {
        hasShaders = lua_toboolean(L, -1) != 0;
    }
    lua_pop(L, 1);
    
    // ========================================================================
    // Parse shader stack configuration (same as accurate renderer)
    // ========================================================================
    std::vector<ActiveShader> active_shaders;
    
    if (hasShaders) {
        // Try new shaderStack format first
        lua_getfield(L, 2, "shaderStack");
        if (lua_istable(L, -1)) {
            // Parse lighting shaders
            lua_getfield(L, -1, "lighting");
            if (lua_istable(L, -1)) {
                parseShaderCategory(L, lua_gettop(L), "lighting", active_shaders);
            }
            lua_pop(L, 1);
            
            // Parse FX shaders
            lua_getfield(L, -1, "fx");
            if (lua_istable(L, -1)) {
                parseShaderCategory(L, lua_gettop(L), "fx", active_shaders);
            }
            lua_pop(L, 1);
        }
        lua_pop(L, 1); // shaderStack
        
        // Fallback to legacy fxStack format
        if (active_shaders.empty()) {
            lua_getfield(L, 2, "fxStack");
            if (lua_istable(L, -1)) {
                lua_getfield(L, -1, "modules");
                if (lua_istable(L, -1)) {
                    parseShaderCategory(L, lua_gettop(L), "fx", active_shaders);
                }
                lua_pop(L, 1);
            }
            lua_pop(L, 1);
        }
    }
    
    // Background color
    unsigned char bgR = 0, bgG = 0, bgB = 0, bgA = 0;
    lua_getfield(L, 2, "backgroundColor");
    if (lua_istable(L, -1)) {
        bgR = (unsigned char)getFieldInteger(L, -1, "r", 0);
        bgG = (unsigned char)getFieldInteger(L, -1, "g", 0);
        bgB = (unsigned char)getFieldInteger(L, -1, "b", 0);
        bgA = (unsigned char)getFieldInteger(L, -1, "a", 0);
    }
    lua_pop(L, 1);
    
    // Parse voxels
    size_t count = lua_rawlen(L, 1);
    lua_createtable(L, 0, 3);
    
    if (count == 0) {
        lua_pushinteger(L, width); lua_setfield(L, -2, "width");
        lua_pushinteger(L, height); lua_setfield(L, -2, "height");
        lua_pushlstring(L, "", 0); lua_setfield(L, -2, "pixels");
        return 1;
    }
    
    std::vector<Voxel> voxels;
    voxels.reserve(count);
    float minX = 1e9f, minY = 1e9f, minZ = 1e9f;
    float maxX = -1e9f, maxY = -1e9f, maxZ = -1e9f;
    
    for (size_t i = 1; i <= count; i++) {
        lua_rawgeti(L, 1, (int)i);
        if (lua_istable(L, -1)) {
            int tblIdx = lua_gettop(L);
            Voxel v{0, 0, 0, 255, 255, 255, 255};
            lua_rawgeti(L, tblIdx, 1);
            bool numeric = lua_isnumber(L, -1) != 0;
            lua_pop(L, 1);
            if (numeric) {
                lua_rawgeti(L, tblIdx, 1); v.x = (float)lua_tonumber(L, -1); lua_pop(L, 1);
                lua_rawgeti(L, tblIdx, 2); v.y = (float)lua_tonumber(L, -1); lua_pop(L, 1);
                lua_rawgeti(L, tblIdx, 3); v.z = (float)lua_tonumber(L, -1); lua_pop(L, 1);
                lua_rawgeti(L, tblIdx, 4); v.r = (unsigned char)lua_tointeger(L, -1); lua_pop(L, 1);
                lua_rawgeti(L, tblIdx, 5); v.g = (unsigned char)lua_tointeger(L, -1); lua_pop(L, 1);
                lua_rawgeti(L, tblIdx, 6); v.b = (unsigned char)lua_tointeger(L, -1); lua_pop(L, 1);
                lua_rawgeti(L, tblIdx, 7); v.a = (unsigned char)lua_tointeger(L, -1); lua_pop(L, 1);
            } else {
                v.x = (float)getNum(L, tblIdx, "x", 0);
                v.y = (float)getNum(L, tblIdx, "y", 0);
                v.z = (float)getNum(L, tblIdx, "z", 0);
                lua_getfield(L, tblIdx, "color");
                if (lua_istable(L, -1)) {
                    v.r = (unsigned char)getFieldInteger(L, -1, "r", 255);
                    v.g = (unsigned char)getFieldInteger(L, -1, "g", 255);
                    v.b = (unsigned char)getFieldInteger(L, -1, "b", 255);
                    v.a = (unsigned char)getFieldInteger(L, -1, "a", 255);
                }
                lua_pop(L, 1);
            }
            if (v.x < minX) minX = v.x; if (v.x > maxX) maxX = v.x;
            if (v.y < minY) minY = v.y; if (v.y > maxY) maxY = v.y;
            if (v.z < minZ) minZ = v.z; if (v.z > maxZ) maxZ = v.z;
            voxels.push_back(v);
        }
        lua_pop(L, 1);
    }
    
    // Build position→voxel-index map for edit mode ID map
    std::unordered_map<uint64_t, uint32_t> posToVoxelIdx;
    if (editMode) {
        for (size_t i = 0; i < voxels.size(); i++) {
            posToVoxelIdx[pack_coords((int)voxels[i].x, (int)voxels[i].y, (int)voxels[i].z)] = (uint32_t)(i + 1);
        }
    }
    
    // Build occupancy map for interior culling
    std::unordered_set<std::string> occ;
    occ.reserve(voxels.size() * 2);
    for (auto& v : voxels) {
        int ix = (int)std::lround(v.x);
        int iy = (int)std::lround(v.y);
        int iz = (int)std::lround(v.z);
        occ.emplace(std::to_string(ix) + "," + std::to_string(iy) + "," + std::to_string(iz));
    }
    
    // Center point and rotation
    // Read custom middlePoint from params if provided, otherwise calculate from bounds
    float midX, midY, midZ;
    lua_getfield(L, 2, "middlePoint");
    if (lua_istable(L, -1)) {
        midX = (float)getNum(L, -1, "x", 0.5f * (minX + maxX));
        midY = (float)getNum(L, -1, "y", 0.5f * (minY + maxY));
        midZ = (float)getNum(L, -1, "z", 0.5f * (minZ + maxZ));
    } else {
        midX = 0.5f * (minX + maxX);
        midY = 0.5f * (minY + maxY);
        midZ = 0.5f * (minZ + maxZ);
    }
    lua_pop(L, 1);
    
    float RX = xRotDeg * (float)PI / 180.f;
    float RY = yRotDeg * (float)PI / 180.f;
    float RZ = zRotDeg * (float)PI / 180.f;
    float cx = std::cos(RX), sx = std::sin(RX);
    float cy = std::cos(RY), sy = std::sin(RY);
    float cz = std::cos(RZ), sz = std::sin(RZ);
    
    // Rotate face normals to determine face visibility
    float rotNormals[6][3];
    for (int f = 0; f < 6; f++) {
        float nx = LOCAL_FACE_NORMALS[f][0];
        float ny = LOCAL_FACE_NORMALS[f][1];
        float nz = LOCAL_FACE_NORMALS[f][2];
        rotateNormal(nx, ny, nz, cx, sx, cy, sy, cz, sz);
        rotNormals[f][0] = nx;
        rotNormals[f][1] = ny;
        rotNormals[f][2] = nz;
    }
    
    // GAP-FREE RENDERING (dynamic expansion approach):
    // When a cube is rotated 45°, diagonal distance between voxel centers increases
    // by sqrt(2) ≈ 1.41x. Since we draw axis-aligned rectangles (not rotated quads),
    // we need to expand the rectangle size based on rotation angle to cover diagonal gaps.
    //
    // Calculate rotation-based expansion factor:
    // At 0° rotation, no extra expansion needed.
    // At 45° rotation, we need ~41% more size to cover the diagonal gap.
    float angleEffect = std::fabs(std::sin(RX * 2.f)) * 0.5f +
                        std::fabs(std::sin(RY * 2.f)) * 0.5f +
                        std::fabs(std::sin(RZ * 2.f)) * 0.25f;
    // Clamp expansion factor between 1.0 and 1.45 (sqrt(2))
    float expansion = 1.0f + std::min(0.45f, angleEffect * 0.5f);
    
    float voxelSize = (scale * expansion) + voxelPadding;  // Dynamic size based on rotation
    float halfSize = voxelSize / 2.0f;
    float screenCX = width * 0.5f;
    float screenCY = height * 0.5f;
    
    // Neighbor offsets (same order as LOCAL_FACE_NORMALS)
    static const int neigh[6][3] = {
        {0, 0, 1}, {0, 0, -1}, {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}
    };
    
    // sRGB utilities for weighted blending
    auto srgbToLinear = [](float c) -> float {
        float normalized = c / 255.0f;
        return normalized * normalized;  // Approximate gamma 2.2
    };
    
    auto linearToSrgb = [](float c) -> unsigned char {
        float result = std::sqrt(c) * 255.0f;
        return (unsigned char)std::min(255.0f, std::max(0.0f, result + 0.5f));
    };
    
    // Process voxels and build draw list with per-voxel visible faces (weighted)
    struct VisibleFaceData {
        int faceIdx;
        float weight;
        float normalX, normalY, normalZ;
    };
    
    struct FastVoxel {
        float screenX, screenY, depth;
        unsigned char r, g, b, a;
        int visibleFace;  // Primary visible face (for backwards compat)
        // Original voxel data for shader processing
        float origX, origY, origZ;
        float normalX, normalY, normalZ;
        // Weighted multi-face data (up to 3 faces visible from any viewpoint)
        int numVisibleFaces;
        VisibleFaceData visibleFaces[3];
        uint32_t voxelId; // Edit-mode: 1-based voxel array index
    };
    std::vector<FastVoxel> drawList;
    drawList.reserve(voxels.size());
    
    for (size_t vi = 0; vi < voxels.size(); vi++) {
        auto& v = voxels[vi];
        int ix = (int)std::lround(v.x);
        int iy = (int)std::lround(v.y);
        int iz = (int)std::lround(v.z);
        
        // Step 1: Find which faces are NOT culled by neighbors
        bool faceUnculled[6] = {false};
        bool hasAnyVisibleFace = false;
        
        for (int f = 0; f < 6; f++) {
            int nx = ix + neigh[f][0];
            int ny = iy + neigh[f][1];
            int nz = iz + neigh[f][2];
            std::string key = std::to_string(nx) + "," + std::to_string(ny) + "," + std::to_string(nz);
            if (occ.find(key) == occ.end()) {
                // No neighbor in this direction - face is potentially visible
                faceUnculled[f] = true;
                hasAnyVisibleFace = true;
            }
        }
        
        if (!hasAnyVisibleFace) continue;  // Skip completely interior voxels
        
        // Step 2: Calculate visibility weights for ALL camera-facing unculled faces
        // Maximum 3 faces can be visible from any single viewpoint (corner view)
        float viewDir[3] = {0, 0, 1};
        
        // Collect all camera-facing faces with their dot products
        struct FaceVis { int face; float dot; };
        FaceVis visFaces[6];
        int numVisFaces = 0;
        float totalDot = 0.0f;
        
        for (int f = 0; f < 6; f++) {
            if (!faceUnculled[f]) continue;
            float dot = rotNormals[f][0] * viewDir[0] + rotNormals[f][1] * viewDir[1] + rotNormals[f][2] * viewDir[2];
            if (dot > 0) {
                visFaces[numVisFaces].face = f;
                visFaces[numVisFaces].dot = dot;
                totalDot += dot;
                numVisFaces++;
            }
        }
        
        // Skip if no faces are camera-facing
        if (numVisFaces == 0) continue;
        
        // Sort by dot product (most visible first)
        for (int i = 0; i < numVisFaces - 1; i++) {
            for (int j = i + 1; j < numVisFaces; j++) {
                if (visFaces[j].dot > visFaces[i].dot) {
                    FaceVis tmp = visFaces[i];
                    visFaces[i] = visFaces[j];
                    visFaces[j] = tmp;
                }
            }
        }
        
        // Primary visible face is the one with highest dot product
        int visibleFace = visFaces[0].face;
        
        // Step 3: Rotate voxel center
        float x = v.x - midX, y = v.y - midY, z = v.z - midZ;
        { float y2 = y * cx - z * sx; float z2 = y * sx + z * cx; y = y2; z = z2; }
        { float x2 = x * cy + z * sy; float z3 = -x * sy + z * cy; x = x2; z = z3; }
        { float x3 = x * cz - y * sz; float y3 = x * sz + y * cz; x = x3; y = y3; }
        
        // Project to screen
        float screenX = screenCX + x * scale;
        float screenY = screenCY + y * scale;
        float depth = z;
        
        FastVoxel fv;
        fv.screenX = screenX;
        fv.screenY = screenY;
        fv.depth = depth;
        fv.r = v.r;
        fv.g = v.g;
        fv.b = v.b;
        fv.a = v.a;
        fv.visibleFace = visibleFace;
        // Store original position and normal for shader processing
        fv.origX = v.x;
        fv.origY = v.y;
        fv.origZ = v.z;
        fv.normalX = rotNormals[visibleFace][0];
        fv.normalY = rotNormals[visibleFace][1];
        fv.normalZ = rotNormals[visibleFace][2];
        
        // Store weighted multi-face data (up to 3 faces)
        fv.numVisibleFaces = std::min(numVisFaces, 3);
        for (int i = 0; i < fv.numVisibleFaces; i++) {
            int f = visFaces[i].face;
            fv.visibleFaces[i].faceIdx = f;
            fv.visibleFaces[i].weight = visFaces[i].dot / totalDot;  // Normalized weight
            fv.visibleFaces[i].normalX = rotNormals[f][0];
            fv.visibleFaces[i].normalY = rotNormals[f][1];
            fv.visibleFaces[i].normalZ = rotNormals[f][2];
        }
        fv.voxelId = (uint32_t)(vi + 1); // 1-based for Lua
        
        drawList.push_back(fv);
    }
    
    // ========================================================================
    // Execute shader pipeline if shaders are enabled
    // Uses BATCHED WEIGHTED MULTI-FACE LIGHTING: build all face data first,
    // execute shaders ONCE on the full batch, then apply weighted blending.
    // This is O(shaders) calls instead of O(voxels × shaders × faces).
    // ========================================================================
    if (hasShaders && !drawList.empty()) {
        // Build native context for shaders
        native_ctx_t ctx;
        memset(&ctx, 0, sizeof(ctx));
        ctx.width = width;
        ctx.height = height;
        ctx.model = nullptr;
        ctx.num_lights = 0;
        ctx.q_view[0] = 0; ctx.q_view[1] = 0; ctx.q_view[2] = 0; ctx.q_view[3] = 1;
        
        // Set model geometry info
        ctx.model_center[0] = midX;
        ctx.model_center[1] = midY;
        ctx.model_center[2] = midZ;
        float sizeX = maxX - minX + 1.f;
        float sizeY = maxY - minY + 1.f;
        float sizeZ = maxZ - minZ + 1.f;
        float diag = std::sqrt(sizeX*sizeX + sizeY*sizeY + sizeZ*sizeZ);
        ctx.model_radius = 0.5f * diag;
        
        // Rotation components for view-to-model transforms
        double rotation_components[6] = {
            (double)cx, (double)sx,
            (double)cy, (double)sy,
            (double)cz, (double)sz
        };
        ctx.rotation = rotation_components;
        
        // ====================================================================
        // BATCHED WEIGHTED MULTI-FACE LIGHTING
        // Phase 1: Build ALL face data into a single batch
        // Phase 2: Execute shaders ONCE on the full batch
        // Phase 3: Apply weighted blending post-shader
        // ====================================================================
        
        // Structure to track which faces belong to which voxel for blending
        struct FaceMapping {
            size_t voxelIdx;      // Index into drawList
            int faceSlot;         // Which visible face (0, 1, 2) or -1 for single-face voxels
            float weight;         // Blending weight (1.0 for single-face voxels)
        };
        
        // Build batch of ALL faces
        std::vector<native_face_data_t> batch_faces;
        std::vector<FaceMapping> face_mappings;
        batch_faces.reserve(drawList.size() * 3);  // Max 3 faces per voxel
        face_mappings.reserve(drawList.size() * 3);
        
        for (size_t vi = 0; vi < drawList.size(); vi++) {
            auto& fv = drawList[vi];
            
            if (fv.numVisibleFaces <= 1) {
                // Single-face voxel: add one face with weight 1.0
                native_face_data_t face;
                face.x = (int)std::lround(fv.origX);
                face.y = (int)std::lround(fv.origY);
                face.z = (int)std::lround(fv.origZ);
                face.face_dir = internalFaceToApiDir(fv.visibleFace);
                face.r = fv.r;
                face.g = fv.g;
                face.b = fv.b;
                face.a = fv.a;
                face.normal[0] = fv.normalX;
                face.normal[1] = fv.normalY;
                face.normal[2] = fv.normalZ;
                face.depth = fv.depth;
                batch_faces.push_back(face);
                
                FaceMapping mapping;
                mapping.voxelIdx = vi;
                mapping.faceSlot = -1;  // Single face
                mapping.weight = 1.0f;
                face_mappings.push_back(mapping);
            } else {
                // Multi-face voxel: add all visible faces
                for (int fi = 0; fi < fv.numVisibleFaces; fi++) {
                    auto& vf = fv.visibleFaces[fi];
                    
                    native_face_data_t face;
                    face.x = (int)std::lround(fv.origX);
                    face.y = (int)std::lround(fv.origY);
                    face.z = (int)std::lround(fv.origZ);
                    face.face_dir = internalFaceToApiDir(vf.faceIdx);
                    face.r = fv.r;  // Original voxel color
                    face.g = fv.g;
                    face.b = fv.b;
                    face.a = fv.a;
                    face.normal[0] = vf.normalX;
                    face.normal[1] = vf.normalY;
                    face.normal[2] = vf.normalZ;
                    face.depth = fv.depth;
                    batch_faces.push_back(face);
                    
                    FaceMapping mapping;
                    mapping.voxelIdx = vi;
                    mapping.faceSlot = fi;
                    mapping.weight = vf.weight;
                    face_mappings.push_back(mapping);
                }
            }
        }
        
        // Execute shaders WITH COLOR ROUTING on the entire batch
        if (!batch_faces.empty()) {
            // Color buffer type for routing
            struct ColorBuffer {
                std::vector<unsigned char> r, g, b, a;
                void resize(size_t n) { r.resize(n); g.resize(n); b.resize(n); a.resize(n); }
                void copyFrom(const std::vector<native_face_data_t>& faces) {
                    resize(faces.size());
                    for (size_t i = 0; i < faces.size(); i++) {
                        r[i] = faces[i].r; g[i] = faces[i].g; 
                        b[i] = faces[i].b; a[i] = faces[i].a;
                    }
                }
                void applyTo(std::vector<native_face_data_t>& faces) const {
                    for (size_t i = 0; i < faces.size() && i < r.size(); i++) {
                        faces[i].r = r[i]; faces[i].g = g[i];
                        faces[i].b = b[i]; faces[i].a = a[i];
                    }
                }
            };
            
            std::unordered_map<std::string, ColorBuffer> outputs;
            ColorBuffer baseColors;
            baseColors.copyFrom(batch_faces);
            outputs["baseColor"] = baseColors;
            ColorBuffer lastColors = baseColors;
            
            auto getEffectiveName = [](const ActiveShader& s) -> std::string {
                if (!s.outputName.empty()) return s.outputName;
                std::string name = s.id;
                size_t dot = name.rfind('.');
                if (dot != std::string::npos) name = name.substr(dot + 1);
                if (!name.empty()) name[0] = toupper(name[0]);
                return "<" + name + "> #" + std::to_string(s.localIndex + 1);
            };
            
            auto getSourceColors = [&](const std::string& sourceName) -> const ColorBuffer& {
                if (sourceName == "last" || sourceName.empty()) return lastColors;
                if (sourceName == "baseColor") return baseColors;
                auto it = outputs.find(sourceName);
                if (it != outputs.end()) return it->second;
                return lastColors;
            };
            
            auto combineAdditive = [](const ColorBuffer& addend, const ColorBuffer& target) -> ColorBuffer {
                ColorBuffer result;
                size_t n = addend.r.size();
                result.resize(n);
                for (size_t i = 0; i < n; i++) {
                    result.r[i] = (unsigned char)std::min(255, (int)addend.r[i] + (int)target.r[i]);
                    result.g[i] = (unsigned char)std::min(255, (int)addend.g[i] + (int)target.g[i]);
                    result.b[i] = (unsigned char)std::min(255, (int)addend.b[i] + (int)target.b[i]);
                    result.a[i] = addend.a[i];
                }
                return result;
            };
            
            // ================================================================
            // TOPOLOGICAL SORT with Kahn's algorithm
            // Dependencies determined by colorInput references
            // Original order used as tiebreaker only
            // ================================================================
            std::unordered_map<std::string, size_t> nameToIdx;
            for (size_t i = 0; i < active_shaders.size(); i++) {
                nameToIdx[getEffectiveName(active_shaders[i])] = i;
            }
            
            std::vector<std::vector<size_t>> adjList(active_shaders.size());
            std::vector<int> inDegree(active_shaders.size(), 0);
            
            for (size_t i = 0; i < active_shaders.size(); i++) {
                const std::string& colorInput = active_shaders[i].colorInput;
                if (!colorInput.empty() && colorInput != "baseColor" && colorInput != "last") {
                    auto it = nameToIdx.find(colorInput);
                    if (it != nameToIdx.end()) {
                        adjList[it->second].push_back(i);
                        inDegree[i]++;
                    }
                }
            }
            
            auto getPriority = [&](size_t idx) -> int {
                const auto& s = active_shaders[idx];
                int catPriority = (s.category == "lighting") ? 0 : 1000;
                return catPriority + s.localIndex;
            };
            
            auto cmp = [&](size_t a, size_t b) { return getPriority(a) < getPriority(b); };
            std::vector<size_t> ready;
            for (size_t i = 0; i < active_shaders.size(); i++) {
                if (inDegree[i] == 0) ready.push_back(i);
            }
            std::sort(ready.begin(), ready.end(), cmp);
            
            std::vector<ActiveShader*> execOrder;
            while (!ready.empty()) {
                size_t curr = ready.front();
                ready.erase(ready.begin());
                execOrder.push_back(&active_shaders[curr]);
                
                for (size_t dep : adjList[curr]) {
                    inDegree[dep]--;
                    if (inDegree[dep] == 0) {
                        ready.push_back(dep);
                        std::sort(ready.begin(), ready.end(), cmp);
                    }
                }
            }
            
            for (auto* sp : execOrder) {
                ActiveShader& s = *sp;
                if (!s.iface || !s.iface->run_voxel_batch) continue;
                
                const ColorBuffer& inputColors = getSourceColors(s.colorInput);
                inputColors.applyTo(batch_faces);
                
                s.iface->run_voxel_batch(s.instance, &ctx, (int)batch_faces.size(), batch_faces.data());
                
                ColorBuffer outputColors;
                outputColors.copyFrom(batch_faces);
                
                if (!s.addTo.empty()) {
                    auto it = outputs.find(s.addTo);
                    if (it != outputs.end()) {
                        outputColors = combineAdditive(outputColors, it->second);
                        outputColors.applyTo(batch_faces);
                    }
                }
                
                outputs[getEffectiveName(s)] = outputColors;
                lastColors = outputColors;
            }
        }
        
        // Apply shader results back to voxels with weighted blending
        // First, initialize accumulators for multi-face voxels
        struct BlendAccum {
            float r, g, b, a;
            bool initialized;
        };
        std::vector<BlendAccum> accumulators(drawList.size(), {0, 0, 0, 0, false});
        
        for (size_t fi = 0; fi < batch_faces.size(); fi++) {
            auto& face = batch_faces[fi];
            auto& mapping = face_mappings[fi];
            auto& fv = drawList[mapping.voxelIdx];
            
            if (mapping.faceSlot < 0) {
                // Single-face voxel: direct assignment
                fv.r = face.r;
                fv.g = face.g;
                fv.b = face.b;
                fv.a = face.a;
            } else {
                // Multi-face voxel: accumulate weighted colors in linear sRGB
                auto& acc = accumulators[mapping.voxelIdx];
                acc.r += srgbToLinear((float)face.r) * mapping.weight;
                acc.g += srgbToLinear((float)face.g) * mapping.weight;
                acc.b += srgbToLinear((float)face.b) * mapping.weight;
                acc.a += (float)face.a * mapping.weight;
                acc.initialized = true;
            }
        }
        
        // Apply accumulated blended colors to multi-face voxels
        for (size_t vi = 0; vi < drawList.size(); vi++) {
            auto& acc = accumulators[vi];
            if (acc.initialized) {
                auto& fv = drawList[vi];
                fv.r = linearToSrgb(acc.r);
                fv.g = linearToSrgb(acc.g);
                fv.b = linearToSrgb(acc.b);
                fv.a = (unsigned char)std::min(255.0f, std::max(0.0f, acc.a + 0.5f));
            }
        }
        
        // Cleanup shader instances
        for (auto& s : active_shaders) {
            if (s.iface && s.instance) {
                s.iface->destroy(s.instance);
            }
        }
    }
    
    // --- Preview voxels (ghost style, single-face, skip shaders) ---
    {
        auto previewVoxels = readPreviewVoxels(L, 2);
        if (!previewVoxels.empty()) {
            // Build combined occupancy for preview interior culling
            std::unordered_set<std::string> combinedOcc = occ;
            for (auto& pv : previewVoxels) {
                combinedOcc.insert(std::to_string((int)pv.x) + "," + std::to_string((int)pv.y) + "," + std::to_string((int)pv.z));
            }

            float viewDir[3] = {0.f, 0.f, 1.f};

            for (auto& pv : previewVoxels) {
                int ix = (int)std::lround(pv.x);
                int iy = (int)std::lround(pv.y);
                int iz = (int)std::lround(pv.z);

                // Find best visible face
                int bestFace = -1;
                float bestDot = 0.f;
                for (int f = 0; f < 6; f++) {
                    std::string key = std::to_string(ix + neigh[f][0]) + "," + std::to_string(iy + neigh[f][1]) + "," + std::to_string(iz + neigh[f][2]);
                    if (combinedOcc.find(key) != combinedOcc.end()) continue;
                    float dot = rotNormals[f][0] * viewDir[0] + rotNormals[f][1] * viewDir[1] + rotNormals[f][2] * viewDir[2];
                    if (dot > bestDot) { bestDot = dot; bestFace = f; }
                }
                if (bestFace < 0) continue;

                float x = pv.x - midX, y = pv.y - midY, z = pv.z - midZ;
                { float y2 = y * cx - z * sx; float z2 = y * sx + z * cx; y = y2; z = z2; }
                { float x2 = x * cy + z * sy; float z3 = -x * sy + z * cy; x = x2; z = z3; }
                { float x3 = x * cz - y * sz; float y3 = x * sz + y * cz; x = x3; y = y3; }

                FastVoxel fv;
                fv.screenX = screenCX + x * scale;
                fv.screenY = screenCY + y * scale;
                fv.depth = z;
                fv.r = pv.r; fv.g = pv.g; fv.b = pv.b; fv.a = pv.a;
                fv.visibleFace = bestFace;
                fv.origX = pv.x; fv.origY = pv.y; fv.origZ = pv.z;
                fv.normalX = rotNormals[bestFace][0];
                fv.normalY = rotNormals[bestFace][1];
                fv.normalZ = rotNormals[bestFace][2];
                fv.numVisibleFaces = 1;
                fv.visibleFaces[0].faceIdx = bestFace;
                fv.visibleFaces[0].weight = 1.0f;
                fv.visibleFaces[0].normalX = rotNormals[bestFace][0];
                fv.visibleFaces[0].normalY = rotNormals[bestFace][1];
                fv.visibleFaces[0].normalZ = rotNormals[bestFace][2];
                fv.voxelId = 0;
                drawList.push_back(fv);
            }
        }
    }
    
    // Sort by depth (painter's algorithm - back to front)
    std::sort(drawList.begin(), drawList.end(), [](const FastVoxel& a, const FastVoxel& b) {
        return a.depth < b.depth;  // Smaller depth (further) drawn first
    });
    
    // Initialize buffer with background
    std::vector<unsigned char> buffer((size_t)width * height * 4, 0);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            size_t off = ((size_t)y * width + x) * 4;
            buffer[off + 0] = bgR;
            buffer[off + 1] = bgG;
            buffer[off + 2] = bgB;
            buffer[off + 3] = bgA;
        }
    }
    
    // Draw rectangles (fast: no polygon rasterization)
    // Use integer coordinates to avoid sub-pixel gaps
    int voxelSizeInt = (int)std::round(voxelSize);
    int halfSizeInt = voxelSizeInt / 2;
    
    // Allocate ID map buffer for edit mode
    std::vector<uint32_t> idMapBuf;
    if (editMode) {
        idMapBuf.resize((size_t)width * height, 0);
    }
    
    for (auto& fv : drawList) {
        // Match Lua: shape.x = screenX - halfSize, shape width/height = voxelSize
        int left = (int)std::round(fv.screenX) - halfSizeInt;
        int top = (int)std::round(fv.screenY) - halfSizeInt;
        int right = left + voxelSizeInt - 1;
        int bottom = top + voxelSizeInt - 1;
        
        // Clip to canvas
        if (left < 0) left = 0;
        if (top < 0) top = 0;
        if (right >= width) right = width - 1;
        if (bottom >= height) bottom = height - 1;
        
        // Encode voxelId + faceIdx for ID map
        uint32_t idVal = 0;
        if (editMode) {
            idVal = (fv.voxelId & 0x00FFFFFFu) | (((uint32_t)fv.visibleFace & 0x7u) << 24);
        }
        
        // Fill rectangle
        for (int py = top; py <= bottom; py++) {
            for (int px = left; px <= right; px++) {
                size_t off = ((size_t)py * width + px) * 4;
                buffer[off + 0] = fv.r;
                buffer[off + 1] = fv.g;
                buffer[off + 2] = fv.b;
                buffer[off + 3] = fv.a;
                if (editMode && fv.voxelId != 0) idMapBuf[(size_t)py * width + px] = idVal;
            }
        }
    }
    
    lua_pushinteger(L, width); lua_setfield(L, -2, "width");
    lua_pushinteger(L, height); lua_setfield(L, -2, "height");
    lua_pushlstring(L, (const char*)buffer.data(), buffer.size()); lua_setfield(L, -2, "pixels");
    if (editMode) {
        lua_pushlstring(L, (const char*)idMapBuf.data(), idMapBuf.size() * sizeof(uint32_t));
        lua_setfield(L, -2, "editIdMap");
    }
    return 1;
}

// ============================================================================
// End Host Helpers
// ============================================================================

// ============================================================================
// BLIT ORCHESTRATOR — Double-buffered background rendering
// ============================================================================

namespace orch {

struct RenderParams {
  int width, height;
  float xRotDeg, yRotDeg, zRotDeg;
  float scale;
  float fovDeg;
  bool orthogonal;
  std::string perspRef;
  float viewportCutoffMultiplier;
  // Background color
  unsigned char bgR, bgG, bgB, bgA;
  // Lighting
  float lightPitch, lightYaw, lightDiffusePct, lightDiameterPct, lightAmbientPct;
  bool rimEnabled;
  unsigned char lightR, lightG, lightB;
  // Shading
  float basicShadeIntensity, basicLightIntensity;
  bool meshMode;
  // Preview voxels (ghost style, per-frame)
  std::vector<Voxel> previewVoxels;
  // Sequence number: incremented on each submit so bg thread can detect stale work
  uint64_t seq;
};

struct VoxelModel {
  std::vector<Voxel> voxels;
  float minX, minY, minZ;
  float maxX, maxY, maxZ;
  float midX, midY, midZ;
};

struct Orchestrator {
  // -- Model data (written by UI thread on sprite change, read by bg thread) --
  std::mutex modelMutex;
  VoxelModel model;
  bool modelReady = false;

  // -- Render params (written by UI thread submit, read by bg thread) --
  std::mutex paramsMutex;
  RenderParams params{};
  bool paramsReady = false;
  uint64_t submitSeq = 0;

  // -- Double buffer (back written by bg thread, front read by UI thread) --
  std::mutex bufferMutex;
  std::vector<unsigned char> backBuffer;
  std::vector<unsigned char> frontBuffer;
  int frontWidth = 0, frontHeight = 0;
  bool renderDone = false;
  uint64_t doneSeq = 0;

  // -- Thread control --
  std::thread bgThread;
  std::condition_variable cv;
  std::mutex cvMutex;
  std::atomic<bool> killing{false};
  std::atomic<bool> stopCurrent{false};
  bool running = false;
};

static Orchestrator g_orch;

// Background thread render function — operates only on owned C++ data
static void orchRenderProc() {
  while (true) {
    // Wait for work
    {
      std::unique_lock<std::mutex> lock(g_orch.cvMutex);
      g_orch.cv.wait(lock, [] {
        return g_orch.paramsReady || g_orch.killing.load();
      });
    }
    if (g_orch.killing.load()) break;

    // Grab the latest params
    RenderParams rp;
    {
      std::lock_guard<std::mutex> lock(g_orch.paramsMutex);
      rp = g_orch.params;
      g_orch.paramsReady = false;
    }

    // Grab model snapshot
    VoxelModel mdl;
    {
      std::lock_guard<std::mutex> lock(g_orch.modelMutex);
      if (!g_orch.modelReady) continue;
      mdl = g_orch.model;  // copy
    }

    g_orch.stopCurrent.store(false);

    // -- Perform the render (dynamic lighting pipeline) --
    int width = rp.width;
    int height = rp.height;
    if (width <= 0 || height <= 0) continue;
    if (mdl.voxels.empty()) {
      // Empty model: produce a cleared buffer
      std::vector<unsigned char> buf((size_t)width * height * 4);
      for (size_t i = 0; i < buf.size(); i += 4) {
        buf[i] = rp.bgR; buf[i+1] = rp.bgG; buf[i+2] = rp.bgB; buf[i+3] = rp.bgA;
      }
      {
        std::lock_guard<std::mutex> lock(g_orch.bufferMutex);
        g_orch.backBuffer = std::move(buf);
        std::swap(g_orch.frontBuffer, g_orch.backBuffer);
        g_orch.frontWidth = width;
        g_orch.frontHeight = height;
        g_orch.renderDone = true;
        g_orch.doneSeq = rp.seq;
      }
      continue;
    }

    float midX = mdl.midX, midY = mdl.midY, midZ = mdl.midZ;
    float sizeX = mdl.maxX - mdl.minX + 1.f;
    float sizeY = mdl.maxY - mdl.minY + 1.f;
    float sizeZ = mdl.maxZ - mdl.minZ + 1.f;
    float maxDim = std::max(sizeX, std::max(sizeY, sizeZ));

    float RX = rp.xRotDeg * (float)PI / 180.f;
    float RY = rp.yRotDeg * (float)PI / 180.f;
    float RZ = rp.zRotDeg * (float)PI / 180.f;
    float cx = std::cos(RX), sx = std::sin(RX);
    float cy = std::cos(RY), sy = std::sin(RY);
    float cz = std::cos(RZ), sz = std::sin(RZ);

    // Camera setup
    bool perspective = (!rp.orthogonal && rp.fovDeg > 0.f);
    float focalLength = 0.f;
    float camDist;
    float fovDeg = rp.fovDeg;
    if (perspective) {
      fovDeg = std::max(5.f, std::min(75.f, fovDeg));
      float warpT = (fovDeg - 5.f) / (75.f - 5.f);
      if (warpT < 0.f) warpT = 0.f; if (warpT > 1.f) warpT = 1.f;
      float amplified = std::pow(warpT, 1.f/3.f);
      const float BASE_NEAR = 1.2f;
      const float FAR_EXTRA = 45.f;
      camDist = maxDim * (BASE_NEAR + (1.f - amplified)*(1.f - amplified) * FAR_EXTRA);
      float fovRad = fovDeg * (float)PI / 180.f;
      focalLength = (height / 2.f) / std::tan(fovRad / 2.f);
    } else {
      camDist = maxDim * 5.f;
    }

    // Voxel size computation
    float voxelSize = rp.scale;
    float vcm = rp.viewportCutoffMultiplier;
    if (vcm <= 0.f) vcm = 2.0f;
    {
      float maxAllowedPixels = std::min(width, height) * vcm;
      if (perspective && focalLength > 1e-6f && maxDim > 0.f) {
        float rcx = cx, rsx = sx, rcy = cy, rsy = sy, rcz = cz, rsz = sz;
        float zMinRot = 1e9f, zMaxRot = -1e9f;
        for (int ix = 0; ix < 2; ++ix) {
          float cxv = (ix == 0) ? mdl.minX : mdl.maxX;
          for (int iy = 0; iy < 2; ++iy) {
            float cyv = (iy == 0) ? mdl.minY : mdl.maxY;
            for (int iz = 0; iz < 2; ++iz) {
              float czv = (iz == 0) ? mdl.minZ : mdl.maxZ;
              float X = cxv - midX, Y = cyv - midY, Z = czv - midZ;
              { float Y2 = Y*rcx - Z*rsx; float Z2 = Y*rsx + Z*rcx; Y = Y2; Z = Z2; }
              { float X2 = X*rcy + Z*rsy; float Z3 = -X*rsy + Z*rcy; X = X2; Z = Z3; }
              { float X3 = X*rcz - Y*rsz; float Y3 = X*rsz + Y*rcz; X = X3; Y = Y3; }
              float worldZ = Z + midZ;
              if (worldZ < zMinRot) zMinRot = worldZ;
              if (worldZ > zMaxRot) zMaxRot = worldZ;
            }
          }
        }
        float cameraZ = midZ + camDist;
        float depthBack = std::max(0.001f, cameraZ - zMinRot);
        float depthFront = std::max(0.001f, cameraZ - zMaxRot);
        float depthMiddle = std::max(0.001f, camDist);
        float depthRef = depthMiddle;
        if (rp.perspRef == "front" || rp.perspRef == "Front") depthRef = depthFront;
        else if (rp.perspRef == "back" || rp.perspRef == "Back") depthRef = depthBack;
        voxelSize = rp.scale * (depthRef / focalLength);
        if (voxelSize * maxDim > maxAllowedPixels) voxelSize = maxAllowedPixels / maxDim;
        if (voxelSize <= 0.f) voxelSize = 1.f;
      } else {
        if (voxelSize < 1.f) voxelSize = 1.f;
        if (voxelSize * maxDim > maxAllowedPixels && maxDim > 0.f)
          voxelSize *= (maxAllowedPixels / (voxelSize * maxDim));
      }
    }

    float threshold = -0.001f;

    // Light direction
    float yawRad = rp.lightYaw * (float)PI / 180.f;
    float pitchRad = rp.lightPitch * (float)PI / 180.f;
    float cosYaw = std::cos(yawRad), sinYaw = std::sin(yawRad);
    float cosPitch = std::cos(pitchRad), sinPitch = std::sin(pitchRad);
    float Lx = cosYaw * cosPitch;
    float Ly = sinPitch;
    float Lz = sinYaw * cosPitch;
    float Lmag = std::sqrt(Lx*Lx + Ly*Ly + Lz*Lz);
    if (Lmag > 1e-6f) { Lx /= Lmag; Ly /= Lmag; Lz /= Lmag; }

    auto rotateVec = [&](float &x, float &y, float &z) {
      float y2 = y*cx - z*sx; float z2 = y*sx + z*cx; y = y2; z = z2;
      float x2 = x*cy + z*sy; float z3 = -x*sy + z*cy; x = x2; z = z3;
      float x3 = x*cz - y*sz; float y3 = x*sz + y*cz; x = x3; y = y3;
    };

    // Face normals in camera space
    struct RFace { float nx, ny, nz; };
    RFace faceNormals[6];
    for (int f = 0; f < 6; f++) {
      float nx = LOCAL_FACE_NORMALS[f][0];
      float ny = LOCAL_FACE_NORMALS[f][1];
      float nz = LOCAL_FACE_NORMALS[f][2];
      rotateVec(nx, ny, nz);
      float m = std::sqrt(nx*nx + ny*ny + nz*nz);
      if (m > 1e-6f) { nx /= m; ny /= m; nz /= m; }
      faceNormals[f] = {nx, ny, nz};
    }

    // Light direction in model space
    auto rotateAxis = [&](float x, float y, float z) -> std::array<float, 3> {
      rotateVec(x, y, z);
      return {x, y, z};
    };
    auto ex = rotateAxis(1, 0, 0);
    auto ey = rotateAxis(0, 1, 0);
    auto ez = rotateAxis(0, 0, 1);
    float Lmx = ex[0]*Lx + ey[0]*Ly + ez[0]*Lz;
    float Lmy = ex[1]*Lx + ey[1]*Ly + ez[1]*Lz;
    float Lmz = ex[2]*Lx + ey[2]*Ly + ez[2]*Lz;
    float Lmm = std::sqrt(Lmx*Lmx + Lmy*Lmy + Lmz*Lmz);
    if (Lmm > 1e-6f) { Lmx /= Lmm; Lmy /= Lmm; Lmz /= Lmm; }

    float diag = std::sqrt(sizeX*sizeX + sizeY*sizeY + sizeZ*sizeZ);
    float modelRadius = 0.5f * diag;
    float dia = std::max(0.f, rp.lightDiameterPct / 100.f);
    float baseRadius = dia * modelRadius;

    float diffNorm = rp.lightDiffusePct / 100.f;
    float exponent = 1.f + (1.f - diffNorm) * 3.f;
    if (exponent < 0.2f) exponent = 0.2f;
    float ambient = rp.lightAmbientPct / 100.f;
    if (ambient > 1.f) ambient = 1.f; if (ambient < 0.f) ambient = 0.f;
    float lightCr = (float)rp.lightR / 255.f;
    float lightCg = (float)rp.lightG / 255.f;
    float lightCb = (float)rp.lightB / 255.f;

    // Build polys
    struct Poly {
      float x[4], y[4], depth;
      unsigned char r, g, b, a;
      float voxelX, voxelY, voxelZ;
      int faceIdx;
    };
    std::vector<Poly> polys;
    polys.reserve(mdl.voxels.size() * 3);
    float screenCX = width * 0.5f;
    float screenCY = height * 0.5f;
    float camZ = midZ + camDist;

    size_t voxelCount = mdl.voxels.size();
    for (size_t vi = 0; vi < voxelCount; ++vi) {
      // Check for cancellation every 2048 voxels
      if ((vi & 2047) == 0 && g_orch.stopCurrent.load()) break;

      const auto &v = mdl.voxels[vi];
      float x = v.x - midX, y = v.y - midY, z = v.z - midZ;
      { float y2 = y*cx - z*sx; float z2 = y*sx + z*cx; y = y2; z = z2; }
      { float x2 = x*cy + z*sy; float z3 = -x*sy + z*cy; x = x2; z = z3; }
      { float x3 = x*cz - y*sz; float y3 = x*sz + y*cz; x = x3; y = y3; }
      float worldX = x + midX, worldY = y + midY, worldZ = z + midZ;

      float vxv, vyv, vzv;
      if (rp.orthogonal) { vxv = 0.f; vyv = 0.f; vzv = 1.f; }
      else {
        vxv = midX - worldX; vyv = midY - worldY; vzv = camZ - worldZ;
        float m = std::sqrt(vxv*vxv + vyv*vyv + vzv*vzv);
        if (m > 1e-5f) { vxv /= m; vyv /= m; vzv /= m; }
      }

      // Radial attenuation
      float mvx = v.x - midX, mvy = v.y - midY, mvz = v.z - midZ;
      float proj = mvx*Lmx + mvy*Lmy + mvz*Lmz;
      float px = mvx - proj*Lmx;
      float py = mvy - proj*Lmy;
      float pz = mvz - proj*Lmz;
      float perp = std::sqrt(px*px + py*py + pz*pz);
      float radial = 1.f;
      if (baseRadius > 1e-6f) {
        radial = 1.f - (perp / baseRadius);
        if (radial < 0.f) radial = 0.f;
      }

      for (int f = 0; f < 6; f++) {
        const auto &fn = faceNormals[f];
        float visDot = fn.nx*vxv + fn.ny*vyv + fn.nz*vzv;
        if (visDot <= threshold) continue;
        float ndotl = fn.nx*Lx + fn.ny*Ly + fn.nz*Lz;
        if (ndotl < 0.f) ndotl = 0.f;
        float diff = std::pow(ndotl, exponent) * radial * diffNorm;
        float rF = v.r * (ambient + diff * lightCr);
        float gF = v.g * (ambient + diff * lightCg);
        float bF = v.b * (ambient + diff * lightCb);
        if (rp.rimEnabled) {
          float ndotv = fn.nz;
          if (ndotv > 0.f) {
            float edge = 1.f - ndotv;
            float t;
            if (edge <= 0.55f) t = 0.f; else if (edge >= 0.95f) t = 1.f;
            else { float tt = (edge - 0.55f) / 0.4f; t = tt*tt*(3.f - 2.f*tt); }
            if (t > 0.f) {
              float rimStrength = 0.6f;
              float rim = rimStrength * t;
              rF += lightCr * rim * 255.f;
              gF += lightCg * rim * 255.f;
              bF += rimStrength * rim * 255.f;
            }
          }
        }
        if (rF < 0) rF = 0; if (rF > 255) rF = 255;
        if (gF < 0) gF = 0; if (gF > 255) gF = 255;
        if (bF < 0) bF = 0; if (bF > 255) bF = 255;

        Poly poly{};
        poly.r = (unsigned char)std::lround(rF);
        poly.g = (unsigned char)std::lround(gF);
        poly.b = (unsigned char)std::lround(bF);
        poly.a = v.a;
        poly.voxelX = v.x; poly.voxelY = v.y; poly.voxelZ = v.z;
        poly.faceIdx = f;
        float avgDepth = 0.f;
        for (int vii = 0; vii < 4; vii++) {
          int vid = FACE_IDX[f][vii] - 1;
          float lx = UNIT_VERTS[vid][0] * voxelSize;
          float ly = UNIT_VERTS[vid][1] * voxelSize;
          float lz = UNIT_VERTS[vid][2] * voxelSize;
          { float y2 = ly*cx - lz*sx; float z2 = ly*sx + lz*cx; ly = y2; lz = z2; }
          { float x2 = lx*cy + lz*sy; float z3 = -lx*sy + lz*cy; lx = x2; lz = z3; }
          { float x3 = lx*cz - ly*sz; float y3 = lx*sz + ly*cz; lx = x3; ly = y3; }
          float wx = worldX * voxelSize + lx;
          float wy = worldY * voxelSize + ly;
          float wz = worldZ + (lz / voxelSize);
          if (perspective) {
            float depth = (camZ - wz);
            if (depth < 0.001f) depth = 0.001f;
            float s = focalLength > 0 ? (focalLength / depth) : 1.f;
            poly.x[vii] = screenCX + (wx - midX * voxelSize) * s;
            poly.y[vii] = screenCY + (wy - midY * voxelSize) * s;
            avgDepth += depth;
          } else {
            poly.x[vii] = screenCX + (wx - midX * voxelSize);
            poly.y[vii] = screenCY + (wy - midY * voxelSize);
            avgDepth += (camZ - worldZ);
          }
        }
        poly.depth = avgDepth / 4.f;
        polys.push_back(poly);
      }
    }

    // --- Preview voxels (ghost style, no lighting) ---
    for (auto& pv : rp.previewVoxels) {
      float x = pv.x - midX, y = pv.y - midY, z = pv.z - midZ;
      { float y2 = y*cx - z*sx; float z2 = y*sx + z*cx; y = y2; z = z2; }
      { float x2 = x*cy + z*sy; float z3 = -x*sy + z*cy; x = x2; z = z3; }
      { float x3 = x*cz - y*sz; float y3 = x*sz + y*cz; x = x3; y = y3; }
      float worldX = x+midX, worldY = y+midY, worldZ = z+midZ;

      float vxv, vyv, vzv;
      if (rp.orthogonal) { vxv=0; vyv=0; vzv=1; }
      else {
        vxv=midX-worldX; vyv=midY-worldY; vzv=camZ-worldZ;
        float m=std::sqrt(vxv*vxv+vyv*vyv+vzv*vzv);
        if(m>1e-5f){vxv/=m;vyv/=m;vzv/=m;}
      }

      for (int f=0;f<6;f++) {
        const auto &fn = faceNormals[f];
        float visDot=fn.nx*vxv+fn.ny*vyv+fn.nz*vzv;
        if(visDot<=threshold) continue;

        Poly poly{};
        poly.r=pv.r; poly.g=pv.g; poly.b=pv.b; poly.a=pv.a;
        poly.voxelX=pv.x; poly.voxelY=pv.y; poly.voxelZ=pv.z;
        poly.faceIdx=f;
        float avgDepth=0.f;
        for(int vii=0;vii<4;vii++){
          int vid=FACE_IDX[f][vii]-1;
          float lx=UNIT_VERTS[vid][0]*voxelSize;
          float ly=UNIT_VERTS[vid][1]*voxelSize;
          float lz=UNIT_VERTS[vid][2]*voxelSize;
          { float y2=ly*cx-lz*sx; float z2=ly*sx+lz*cx; ly=y2; lz=z2; }
          { float x2=lx*cy+lz*sy; float z3=-lx*sy+lz*cy; lx=x2; lz=z3; }
          { float x3=lx*cz-ly*sz; float y3=lx*sz+ly*cz; lx=x3; ly=y3; }
          float wx=worldX*voxelSize+lx;
          float wy=worldY*voxelSize+ly;
          float wz=worldZ+(lz/voxelSize);
          if(perspective){
            float depth=(camZ-wz); if(depth<0.001f)depth=0.001f;
            float s=focalLength>0?(focalLength/depth):1.f;
            poly.x[vii]=screenCX+(wx-midX*voxelSize)*s;
            poly.y[vii]=screenCY+(wy-midY*voxelSize)*s;
            avgDepth+=depth;
          } else {
            poly.x[vii]=screenCX+(wx-midX*voxelSize);
            poly.y[vii]=screenCY+(wy-midY*voxelSize);
            avgDepth+=(camZ-worldZ);
          }
        }
        poly.depth=avgDepth/4.f;
        polys.push_back(poly);
      }
    }

    // Check cancellation before sorting
    if (g_orch.stopCurrent.load()) continue;

    // Sort back-to-front
    std::sort(polys.begin(), polys.end(), [](const Poly &a, const Poly &b) {
      constexpr float DEPTH_EPS = 0.0001f;
      if (std::abs(a.depth - b.depth) > DEPTH_EPS) return a.depth > b.depth;
      if (a.voxelZ != b.voxelZ) return a.voxelZ < b.voxelZ;
      if (a.voxelY != b.voxelY) return a.voxelY < b.voxelY;
      if (a.voxelX != b.voxelX) return a.voxelX < b.voxelX;
      return a.faceIdx < b.faceIdx;
    });

    // Rasterize to backbuffer
    size_t bufSize = (size_t)width * height * 4;
    std::vector<unsigned char> buf(bufSize);
    for (size_t i = 0; i < bufSize; i += 4) {
      buf[i] = rp.bgR; buf[i+1] = rp.bgG; buf[i+2] = rp.bgB; buf[i+3] = rp.bgA;
    }

    for (size_t pi = 0; pi < polys.size(); ++pi) {
      if ((pi & 4095) == 0 && g_orch.stopCurrent.load()) break;
      FacePoly fp;
      for (int i = 0; i < 4; i++) { fp.x[i] = polys[pi].x[i]; fp.y[i] = polys[pi].y[i]; }
      fp.depth = polys[pi].depth;
      fp.r = polys[pi].r; fp.g = polys[pi].g; fp.b = polys[pi].b; fp.a = polys[pi].a;
      rasterQuad(fp, width, height, buf);
    }

    // Check cancellation before publishing
    if (g_orch.stopCurrent.load()) continue;

    // Publish result: swap into front buffer
    {
      std::lock_guard<std::mutex> lock(g_orch.bufferMutex);
      g_orch.backBuffer = std::move(buf);
      std::swap(g_orch.frontBuffer, g_orch.backBuffer);
      g_orch.frontWidth = width;
      g_orch.frontHeight = height;
      g_orch.renderDone = true;
      g_orch.doneSeq = rp.seq;
    }

    // Check if a newer submit arrived while we were rendering
    {
      std::lock_guard<std::mutex> lock(g_orch.paramsMutex);
      if (g_orch.paramsReady) {
        // Loop again immediately — the cv.wait at top will pass through
        continue;
      }
    }
  }
}

// orch_start() — spawn background thread
static int l_orch_start(lua_State* L) {
  if (g_orch.running) {
    lua_pushboolean(L, 1);
    return 1;
  }
  g_orch.killing.store(false);
  g_orch.stopCurrent.store(false);
  g_orch.renderDone = false;
  g_orch.paramsReady = false;
  g_orch.submitSeq = 0;
  g_orch.doneSeq = 0;
  g_orch.bgThread = std::thread(orchRenderProc);
  g_orch.running = true;
  lua_pushboolean(L, 1);
  return 1;
}

// orch_stop() — join background thread
static int l_orch_stop(lua_State* L) {
  if (!g_orch.running) {
    lua_pushboolean(L, 1);
    return 1;
  }
  g_orch.killing.store(true);
  g_orch.stopCurrent.store(true);
  g_orch.cv.notify_one();
  if (g_orch.bgThread.joinable()) {
    g_orch.bgThread.join();
  }
  g_orch.running = false;
  g_orch.renderDone = false;
  g_orch.paramsReady = false;
  g_orch.modelReady = false;
  lua_pushboolean(L, 1);
  return 1;
}

// orch_set_model(voxelTable) — copy voxel data from Lua table into C++ storage
static int l_orch_set_model(lua_State* L) {
  if (!lua_istable(L, 1)) {
    lua_pushboolean(L, 0);
    lua_pushstring(L, "expected voxel table");
    return 2;
  }
  size_t vcount = lua_rawlen(L, 1);
  VoxelModel mdl;
  mdl.voxels.reserve(vcount);
  mdl.minX = 1e9f; mdl.minY = 1e9f; mdl.minZ = 1e9f;
  mdl.maxX = -1e9f; mdl.maxY = -1e9f; mdl.maxZ = -1e9f;

  for (size_t i = 1; i <= vcount; i++) {
    lua_rawgeti(L, 1, (int)i);
    if (lua_istable(L, -1)) {
      int t = lua_gettop(L);
      Voxel v{0, 0, 0, 255, 255, 255, 255};
      // Detect numeric array vs named fields
      lua_rawgeti(L, t, 1);
      bool numeric = lua_isnumber(L, -1) != 0;
      lua_pop(L, 1);
      if (numeric) {
        lua_rawgeti(L, t, 1); v.x = (float)lua_tonumber(L, -1); lua_pop(L, 1);
        lua_rawgeti(L, t, 2); v.y = (float)lua_tonumber(L, -1); lua_pop(L, 1);
        lua_rawgeti(L, t, 3); v.z = (float)lua_tonumber(L, -1); lua_pop(L, 1);
        lua_rawgeti(L, t, 4); v.r = (unsigned char)lua_tointeger(L, -1); lua_pop(L, 1);
        lua_rawgeti(L, t, 5); v.g = (unsigned char)lua_tointeger(L, -1); lua_pop(L, 1);
        lua_rawgeti(L, t, 6); v.b = (unsigned char)lua_tointeger(L, -1); lua_pop(L, 1);
        lua_rawgeti(L, t, 7); v.a = (unsigned char)lua_tointeger(L, -1); lua_pop(L, 1);
      } else {
        v.x = (float)getNum(L, t, "x", 0);
        v.y = (float)getNum(L, t, "y", 0);
        v.z = (float)getNum(L, t, "z", 0);
        lua_getfield(L, t, "color");
        if (lua_istable(L, -1)) {
          v.r = (unsigned char)getFieldInteger(L, -1, "r", 255);
          v.g = (unsigned char)getFieldInteger(L, -1, "g", 255);
          v.b = (unsigned char)getFieldInteger(L, -1, "b", 255);
          v.a = (unsigned char)getFieldInteger(L, -1, "a", 255);
        }
        lua_pop(L, 1);
      }
      if (v.x < mdl.minX) mdl.minX = v.x; if (v.x > mdl.maxX) mdl.maxX = v.x;
      if (v.y < mdl.minY) mdl.minY = v.y; if (v.y > mdl.maxY) mdl.maxY = v.y;
      if (v.z < mdl.minZ) mdl.minZ = v.z; if (v.z > mdl.maxZ) mdl.maxZ = v.z;
      mdl.voxels.push_back(v);
    }
    lua_pop(L, 1);
  }

  // Support custom middlePoint from optional second arg
  if (lua_istable(L, 2)) {
    mdl.midX = (float)getNum(L, 2, "x", 0.5f * (mdl.minX + mdl.maxX));
    mdl.midY = (float)getNum(L, 2, "y", 0.5f * (mdl.minY + mdl.maxY));
    mdl.midZ = (float)getNum(L, 2, "z", 0.5f * (mdl.minZ + mdl.maxZ));
  } else {
    mdl.midX = 0.5f * (mdl.minX + mdl.maxX);
    mdl.midY = 0.5f * (mdl.minY + mdl.maxY);
    mdl.midZ = 0.5f * (mdl.minZ + mdl.maxZ);
  }

  {
    std::lock_guard<std::mutex> lock(g_orch.modelMutex);
    g_orch.model = std::move(mdl);
    g_orch.modelReady = true;
  }

  lua_pushboolean(L, 1);
  lua_pushinteger(L, (lua_Integer)vcount);
  return 2;
}

// orch_submit(paramsTable) — copy render params, signal background thread (non-blocking)
static int l_orch_submit(lua_State* L) {
  if (!lua_istable(L, 1)) {
    lua_pushboolean(L, 0);
    return 1;
  }

  RenderParams rp{};
  rp.width = (int)getNum(L, 1, "width", 200);
  rp.height = (int)getNum(L, 1, "height", 200);
  rp.xRotDeg = (float)getNum(L, 1, "xRotation", 0.0);
  rp.yRotDeg = (float)getNum(L, 1, "yRotation", 0.0);
  rp.zRotDeg = (float)getNum(L, 1, "zRotation", 0.0);
  rp.scale = (float)getNum(L, 1, "scale", -12345.0);
  if (rp.scale < 0) rp.scale = (float)getNum(L, 1, "scaleLevel", 1.0);
  if (rp.scale <= 0) rp.scale = 1.f;
  rp.fovDeg = (float)getNum(L, 1, "fovDegrees", 0.0);
  lua_getfield(L, 1, "orthogonal");
  rp.orthogonal = lua_toboolean(L, -1) != 0;
  lua_pop(L, 1);
  rp.viewportCutoffMultiplier = (float)getNum(L, 1, "viewportCutoffMultiplier", 2.0);
  lua_getfield(L, 1, "perspectiveScaleRef");
  if (lua_isstring(L, -1)) rp.perspRef = lua_tostring(L, -1);
  else rp.perspRef = "middle";
  lua_pop(L, 1);

  // Background color
  lua_getfield(L, 1, "backgroundColor");
  if (lua_istable(L, -1)) {
    rp.bgR = (unsigned char)getFieldInteger(L, -1, "r", 0);
    rp.bgG = (unsigned char)getFieldInteger(L, -1, "g", 0);
    rp.bgB = (unsigned char)getFieldInteger(L, -1, "b", 0);
    rp.bgA = (unsigned char)getFieldInteger(L, -1, "a", 0);
  }
  lua_pop(L, 1);

  // Lighting params
  lua_getfield(L, 1, "lighting");
  if (lua_istable(L, -1)) {
    rp.lightPitch = (float)getNum(L, -1, "pitch", 0.0);
    rp.lightYaw = (float)getNum(L, -1, "yaw", 0.0);
    rp.lightDiffusePct = (float)getNum(L, -1, "diffuse", 60.0);
    rp.lightDiameterPct = (float)getNum(L, -1, "diameter", 100.0);
    rp.lightAmbientPct = (float)getNum(L, -1, "ambient", 30.0);
    lua_getfield(L, -1, "rimEnabled");
    rp.rimEnabled = lua_toboolean(L, -1) != 0;
    lua_pop(L, 1);
    lua_getfield(L, -1, "lightColor");
    if (lua_istable(L, -1)) {
      rp.lightR = (unsigned char)getFieldInteger(L, -1, "r", 255);
      rp.lightG = (unsigned char)getFieldInteger(L, -1, "g", 255);
      rp.lightB = (unsigned char)getFieldInteger(L, -1, "b", 255);
    } else {
      rp.lightR = 255; rp.lightG = 255; rp.lightB = 255;
    }
    lua_pop(L, 1);
  } else {
    rp.lightDiffusePct = 60.f; rp.lightAmbientPct = 30.f;
    rp.lightDiameterPct = 100.f;
    rp.lightR = 255; rp.lightG = 255; rp.lightB = 255;
  }
  lua_pop(L, 1);

  rp.basicShadeIntensity = (float)getNum(L, 1, "basicShadeIntensity", 50.0);
  rp.basicLightIntensity = (float)getNum(L, 1, "basicLightIntensity", 50.0);
  lua_getfield(L, 1, "mesh");
  rp.meshMode = lua_toboolean(L, -1) != 0;
  lua_pop(L, 1);

  // Preview voxels
  rp.previewVoxels = readPreviewVoxels(L, 1);

  // Cancel any in-progress render
  g_orch.stopCurrent.store(true);

  {
    std::lock_guard<std::mutex> lock(g_orch.paramsMutex);
    rp.seq = ++g_orch.submitSeq;
    g_orch.params = rp;
    g_orch.paramsReady = true;
  }
  g_orch.cv.notify_one();

  lua_pushboolean(L, 1);
  return 1;
}

// orch_get_pixels() — returns pixels string + width + height if render done, else nil
// Usage: local pixels, w, h = native.orch_get_pixels()
//        if pixels then img.bytes = pixels end
static int l_orch_get_pixels(lua_State* L) {
  std::lock_guard<std::mutex> lock(g_orch.bufferMutex);
  if (!g_orch.renderDone) {
    lua_pushnil(L);
    return 1;
  }
  g_orch.renderDone = false;
  lua_pushlstring(L, (const char*)g_orch.frontBuffer.data(), g_orch.frontBuffer.size());
  lua_pushinteger(L, g_orch.frontWidth);
  lua_pushinteger(L, g_orch.frontHeight);
  return 3;
}

// orch_status() — returns { running, modelReady, renderDone, submitSeq, doneSeq }
static int l_orch_status(lua_State* L) {
  lua_createtable(L, 0, 5);
  lua_pushboolean(L, g_orch.running ? 1 : 0);
  lua_setfield(L, -2, "running");
  lua_pushboolean(L, g_orch.modelReady ? 1 : 0);
  lua_setfield(L, -2, "modelReady");
  {
    std::lock_guard<std::mutex> lock(g_orch.bufferMutex);
    lua_pushboolean(L, g_orch.renderDone ? 1 : 0);
    lua_setfield(L, -2, "renderDone");
  }
  lua_pushinteger(L, (lua_Integer)g_orch.submitSeq);
  lua_setfield(L, -2, "submitSeq");
  lua_pushinteger(L, (lua_Integer)g_orch.doneSeq);
  lua_setfield(L, -2, "doneSeq");
  return 1;
}

// ============================================================================
// End Blit Orchestrator
// ============================================================================

} // namespace orch

// =============================================================================
// UI OVERLAY RENDERER — render small set of UI voxels onto transparent buffer
// Uses pre-computed camera/projection params from the main render pass.
//
// Lua signature: render_overlay(uiVoxels, params) -> result table
//   uiVoxels: array of {x,y,z, color={r,g,b,a}, style="solid"|"wireframe"|"ghost"}
//   params: {width, height, voxelSize, xRotation, yRotation, zRotation,
//            middlePoint={x,y,z}, orthogonal, fovDegrees, _cameraDistance}
//   result: {width, height, pixels}
// =============================================================================
static int l_render_overlay(lua_State* L) {
  if (!lua_istable(L, 1) || !lua_istable(L, 2)) {
    lua_pushnil(L);
    lua_pushstring(L, "expected (uiVoxels, params) tables");
    return 2;
  }

  int width  = (int)getNum(L, 2, "width", 200);
  int height = (int)getNum(L, 2, "height", 200);
  float voxelSize = (float)getNum(L, 2, "voxelSize", 1.0);
  if (voxelSize <= 0.f) voxelSize = 1.f;

  float xRotDeg = (float)getNum(L, 2, "xRotation", 0.0);
  float yRotDeg = (float)getNum(L, 2, "yRotation", 0.0);
  float zRotDeg = (float)getNum(L, 2, "zRotation", 0.0);

  lua_getfield(L, 2, "orthogonal");
  bool orthogonal = lua_toboolean(L, -1) != 0;
  lua_pop(L, 1);

  float fovDeg = (float)getNum(L, 2, "fovDegrees", 0.0);
  float camDist = (float)getNum(L, 2, "_cameraDistance", 100.0);

  float midX = 0.f, midY = 0.f, midZ = 0.f;
  lua_getfield(L, 2, "middlePoint");
  if (lua_istable(L, -1)) {
    midX = (float)getNum(L, -1, "x", 0.0);
    midY = (float)getNum(L, -1, "y", 0.0);
    midZ = (float)getNum(L, -1, "z", 0.0);
  }
  lua_pop(L, 1);

  // Compute rotation trig
  float RX = xRotDeg * (float)PI / 180.f;
  float RY = yRotDeg * (float)PI / 180.f;
  float RZ = zRotDeg * (float)PI / 180.f;
  float cx_ = std::cos(RX), sx_ = std::sin(RX);
  float cy_ = std::cos(RY), sy_ = std::sin(RY);
  float cz_ = std::cos(RZ), sz_ = std::sin(RZ);

  // Camera setup
  bool perspective = (!orthogonal && fovDeg > 0.f);
  float focalLength = 0.f;
  float screenCX = width * 0.5f;
  float screenCY = height * 0.5f;
  float camZ = midZ + camDist;
  if (perspective) {
    fovDeg = std::max(5.f, std::min(75.f, fovDeg));
    float fovRad = fovDeg * (float)PI / 180.f;
    focalLength = (height / 2.f) / std::tan(fovRad / 2.f);
  }

  // Global view vector for face visibility
  float gvz = 1.f; // orthographic default
  if (perspective && camDist > 1e-6f) {
    gvz = 1.f; // camera looks along -Z toward model center → view = (0,0,1)
  }

  // Parse UI voxels
  size_t count = lua_rawlen(L, 1);
  lua_createtable(L, 0, 3);
  if (count == 0) {
    lua_pushinteger(L, width);  lua_setfield(L, -2, "width");
    lua_pushinteger(L, height); lua_setfield(L, -2, "height");
    lua_pushlstring(L, "", 0);  lua_setfield(L, -2, "pixels");
    return 1;
  }

  struct UIVoxel {
    float x, y, z;
    unsigned char r, g, b, a;
    int style; // 0=solid, 1=wireframe, 2=ghost
  };
  std::vector<UIVoxel> uiVoxels;
  uiVoxels.reserve(count);
  for (size_t i = 1; i <= count; i++) {
    lua_rawgeti(L, 1, (int)i);
    if (lua_istable(L, -1)) {
      int tbl = lua_gettop(L);
      UIVoxel uv{0,0,0, 255,255,255,128, 2};
      uv.x = (float)getNum(L, tbl, "x", 0.0);
      uv.y = (float)getNum(L, tbl, "y", 0.0);
      uv.z = (float)getNum(L, tbl, "z", 0.0);
      lua_getfield(L, tbl, "color");
      if (lua_istable(L, -1)) {
        uv.r = (unsigned char)getFieldInteger(L, -1, "r", 255);
        uv.g = (unsigned char)getFieldInteger(L, -1, "g", 255);
        uv.b = (unsigned char)getFieldInteger(L, -1, "b", 255);
        uv.a = (unsigned char)getFieldInteger(L, -1, "a", 128);
      }
      lua_pop(L, 1);
      lua_getfield(L, tbl, "style");
      if (lua_isstring(L, -1)) {
        const char* s = lua_tostring(L, -1);
        if (s) {
          if (s[0] == 's') uv.style = 0;      // solid
          else if (s[0] == 'w') uv.style = 1; // wireframe
          else uv.style = 2;                    // ghost (default)
        }
      }
      lua_pop(L, 1);
      uiVoxels.push_back(uv);
    }
    lua_pop(L, 1);
  }

  // Create face polys (same approach as render_basic but simpler)
  std::vector<FacePoly> polys;
  polys.reserve(uiVoxels.size() * 6);
  float threshold = -0.001f;

  for (size_t vi = 0; vi < uiVoxels.size(); vi++) {
    auto& v = uiVoxels[vi];
    float x = v.x - midX, y = v.y - midY, z = v.z - midZ;
    // Rotate center
    { float y2 = y*cx_ - z*sx_; float z2 = y*sx_ + z*cx_; y = y2; z = z2; }
    { float x2 = x*cy_ + z*sy_; float z3 = -x*sy_ + z*cy_; x = x2; z = z3; }
    { float x3 = x*cz_ - y*sz_; float y3 = x*sz_ + y*cz_; x = x3; y = y3; }
    float worldX = x + midX;
    float worldY = y + midY;
    float worldZ = z + midZ;

    for (int f = 0; f < 6; f++) {
      float nx = LOCAL_FACE_NORMALS[f][0];
      float ny = LOCAL_FACE_NORMALS[f][1];
      float nz = LOCAL_FACE_NORMALS[f][2];
      rotateNormal(nx, ny, nz, cx_, sx_, cy_, sy_, cz_, sz_);

      // View direction for face visibility
      float vxv = 0.f, vyv = 0.f, vzv = gvz;
      if (perspective) {
        vxv = midX - worldX;
        vyv = midY - worldY;
        vzv = camZ - worldZ;
        float pmag = std::sqrt(vxv*vxv + vyv*vyv + vzv*vzv);
        if (pmag > 1e-5f) { vxv /= pmag; vyv /= pmag; vzv /= pmag; }
      }
      if ((nx*vxv + ny*vyv + nz*vzv) <= threshold) continue;

      // Compute screen vertices
      FacePoly poly{};
      // Determine alpha based on style
      unsigned char alpha = v.a;
      if (v.style == 2) { // ghost
        alpha = (unsigned char)std::max(1, (int)(v.a * 0.5f));
      }
      poly.r = v.r; poly.g = v.g; poly.b = v.b; poly.a = alpha;
      poly.voxelX = v.x; poly.voxelY = v.y; poly.voxelZ = v.z;
      poly.faceIdx = f;
      poly.voxelId = 0; // No ID map for overlay

      float avgDepth = 0.f;
      for (int i = 0; i < 4; i++) {
        int vidx = FACE_IDX[f][i] - 1;
        float lx = UNIT_VERTS[vidx][0] * voxelSize;
        float ly = UNIT_VERTS[vidx][1] * voxelSize;
        float lz = UNIT_VERTS[vidx][2] * voxelSize;
        rotateNormal(lx, ly, lz, cx_, sx_, cy_, sy_, cz_, sz_);
        float wx = worldX * voxelSize + lx;
        float wy = worldY * voxelSize + ly;
        float wzLocal = worldZ + (lz / voxelSize);

        float depth;
        float s = 1.0f;
        if (perspective) {
          depth = camZ - wzLocal;
          if (depth < 0.001f) depth = 0.001f;
          s = (focalLength > 0.f) ? (focalLength / depth) : 1.0f;
        } else {
          depth = camZ - wzLocal;
        }
        poly.x[i] = screenCX + (wx - midX * voxelSize) * s;
        poly.y[i] = screenCY + (wy - midY * voxelSize) * s;
        avgDepth += depth;
      }
      poly.depth = avgDepth / 4.f;
      polys.push_back(poly);
    }
  }

  // Sort far to near (painter's algorithm)
  std::sort(polys.begin(), polys.end(), [](const FacePoly& a, const FacePoly& b) {
    constexpr float EPS = 0.0001f;
    if (std::abs(a.depth - b.depth) > EPS) return a.depth > b.depth;
    if (a.voxelZ != b.voxelZ) return a.voxelZ < b.voxelZ;
    if (a.voxelY != b.voxelY) return a.voxelY < b.voxelY;
    if (a.voxelX != b.voxelX) return a.voxelX < b.voxelX;
    return a.faceIdx < b.faceIdx;
  });

  // Rasterize onto transparent buffer (RGBA all zeros = transparent)
  std::vector<unsigned char> buffer((size_t)width * height * 4, 0);
  for (auto& p : polys) {
    rasterQuad(p, width, height, buffer, nullptr);
  }

  // Wireframe pass: for wireframe/ghost style voxels, draw edges on top
  // (We handle this by drawing edge-only polys in a second pass for those styles)
  // Implementation: iterate uiVoxels, for style=wireframe draw edges only,
  // for ghost the filled pass above already ran, now add edge outlines.
  for (size_t vi = 0; vi < uiVoxels.size(); vi++) {
    auto& v = uiVoxels[vi];
    if (v.style == 0) continue; // solid: no extra edges needed

    float x = v.x - midX, y = v.y - midY, z = v.z - midZ;
    { float y2 = y*cx_ - z*sx_; float z2 = y*sx_ + z*cx_; y = y2; z = z2; }
    { float x2 = x*cy_ + z*sy_; float z3 = -x*sy_ + z*cy_; x = x2; z = z3; }
    { float x3 = x*cz_ - y*sz_; float y3 = x*sz_ + y*cz_; x = x3; y = y3; }
    float worldX = x + midX;
    float worldY = y + midY;
    float worldZ = z + midZ;

    // Project all 8 vertices
    float sx_arr[8], sy_arr[8];
    for (int i = 0; i < 8; i++) {
      float lx = UNIT_VERTS[i][0] * voxelSize;
      float ly = UNIT_VERTS[i][1] * voxelSize;
      float lz = UNIT_VERTS[i][2] * voxelSize;
      rotateNormal(lx, ly, lz, cx_, sx_, cy_, sy_, cz_, sz_);
      float wx = worldX * voxelSize + lx;
      float wy = worldY * voxelSize + ly;
      float wzLocal = worldZ + (lz / voxelSize);

      float s = 1.0f;
      if (perspective) {
        float depth = camZ - wzLocal;
        if (depth < 0.001f) depth = 0.001f;
        s = (focalLength > 0.f) ? (focalLength / depth) : 1.0f;
      }
      sx_arr[i] = screenCX + (wx - midX * voxelSize) * s;
      sy_arr[i] = screenCY + (wy - midY * voxelSize) * s;
    }

    // Draw visible face edges using Bresenham-like line drawing
    unsigned char edgeAlpha = std::min((int)v.a + 40, 255);
    for (int f = 0; f < 6; f++) {
      float nx = LOCAL_FACE_NORMALS[f][0];
      float ny = LOCAL_FACE_NORMALS[f][1];
      float nz = LOCAL_FACE_NORMALS[f][2];
      rotateNormal(nx, ny, nz, cx_, sx_, cy_, sy_, cz_, sz_);
      float vxv = 0.f, vyv = 0.f, vzv = gvz;
      if (perspective) {
        vxv = midX - worldX; vyv = midY - worldY; vzv = camZ - worldZ;
        float pmag = std::sqrt(vxv*vxv + vyv*vyv + vzv*vzv);
        if (pmag > 1e-5f) { vxv /= pmag; vyv /= pmag; vzv /= pmag; }
      }
      if ((nx*vxv + ny*vyv + nz*vzv) <= threshold) continue;

      for (int e = 0; e < 4; e++) {
        int i0 = FACE_IDX[f][e] - 1;
        int i1 = FACE_IDX[f][(e + 1) & 3] - 1;
        float ax = sx_arr[i0], ay = sy_arr[i0];
        float bx = sx_arr[i1], by = sy_arr[i1];
        int steps = std::max((int)std::abs(bx - ax), (int)std::abs(by - ay));
        if (steps < 1) steps = 1;
        for (int s = 0; s <= steps; s++) {
          float t = (float)s / steps;
          int px = (int)(ax + (bx - ax) * t + 0.5f);
          int py = (int)(ay + (by - ay) * t + 0.5f);
          if (px >= 0 && px < width && py >= 0 && py < height) {
            size_t off = ((size_t)py * width + px) * 4;
            buffer[off + 0] = v.r;
            buffer[off + 1] = v.g;
            buffer[off + 2] = v.b;
            buffer[off + 3] = edgeAlpha;
          }
        }
      }
    }
  }

  // Return result
  lua_pushinteger(L, width);  lua_setfield(L, -2, "width");
  lua_pushinteger(L, height); lua_setfield(L, -2, "height");
  lua_pushlstring(L, (const char*)buffer.data(), buffer.size());
  lua_setfield(L, -2, "pixels");
  return 1;
}

// Forward declaration: defined below FUNCS (needed for FUNCS table reference)
extern "C" int asevoxel_schedule_unload(lua_State* L);

// BUG-009: forward declarations for the native process-spawn helpers, defined
// below FUNCS (need windows.h, included further down). See their definitions
// for the full rationale.
extern "C" int asevoxel_spawn_capture(lua_State* L);
extern "C" int asevoxel_spawn_detached(lua_State* L);

static const luaL_Reg FUNCS[] = {
  {"transform_voxel", l_transform_voxel},
  {"calculate_face_visibility", l_calculate_face_visibility},
  {"render_basic", l_render_basic},
  {"render_fast", l_render_fast},
  {"render_overlay", l_render_overlay},
  {"scan_shaders", l_scan_shaders},
  {"unload_shaders", l_unload_shaders},
  {"scan_shaders_debug", l_scan_shaders_debug},
  {"try_load_shader_dll", l_try_load_shader_dll},
  {"set_debug_shader_params", l_set_debug_shader_params},
  {"set_debug_logging", l_set_debug_logging},
  {"get_shader_info", l_get_shader_info},
  // Phase 2.3/3: New shader stack renderer with proper shaderStack support
  {"render_shader_stack", l_render_shader_stack},
  // Legacy render_stack (forwards to new implementation)
  {"render_stack", l_render_shader_stack},  // Now uses the same implementation
  // Schedules FreeLibrary on a background thread so Lua can release the DLL
  // without the file lock preventing extension uninstall/update on Windows.
  {"schedule_unload", asevoxel_schedule_unload},
  // BUG-009: spawn a child process without requesting a console at all
  // (CreateProcess + DETACHED_PROCESS), bypassing os.execute()/io.popen()'s
  // implicit cmd.exe + console-allocation path entirely. See the
  // definitions below for the full story.
  {"spawn_capture", asevoxel_spawn_capture},
  {"spawn_detached", asevoxel_spawn_detached},
  // --------------------------------------------------------------------------
  // BLIT ORCHESTRATOR — double-buffered background rendering
  // --------------------------------------------------------------------------
  {"orch_start", orch::l_orch_start},
  {"orch_stop", orch::l_orch_stop},
  {"orch_set_model", orch::l_orch_set_model},
  {"orch_submit", orch::l_orch_submit},
  {"orch_get_pixels", orch::l_orch_get_pixels},
  {"orch_status", orch::l_orch_status},
  // --------------------------------------------------------------------------
  // DYNAMIC LIGHTING RENDERER (Lambert + exponent, ambient, radial falloff,
  // rim lighting)
  // --------------------------------------------------------------------------
  {"render_dynamic", [](lua_State* L)->int {
      if(!lua_istable(L,1)||!lua_istable(L,2)){
        lua_pushnil(L); lua_pushstring(L,"expected (voxels, params)"); return 2;
      }
      int width  =(int)getNum(L,2,"width",200);

      int height =(int)getNum(L,2,"height",200);
      float scale=(float)getNum(L,2,"scale",-12345.0f);
      if (scale < 0) scale=(float)getNum(L,2,"scaleLevel",1.0f);
      if (scale <= 0) scale = 1.f;
    float viewportCutoffMultiplier = (float)getNum(L,2,"viewportCutoffMultiplier",2.0f);
    if (viewportCutoffMultiplier <= 0.f) viewportCutoffMultiplier = 2.0f;
      float xRotDeg=(float)getNum(L,2,"xRotation",0.0f);
      float yRotDeg=(float)getNum(L,2,"yRotation",0.0f);
      float zRotDeg=(float)getNum(L,2,"zRotation",0.0f);
      float fovDeg =(float)getNum(L,2,"fovDegrees",0.0f);
      lua_getfield(L,2,"orthogonal"); bool orth = lua_toboolean(L,-1)!=0; lua_pop(L,1);
      std::string perspRef="middle";
      lua_getfield(L,2,"perspectiveScaleRef");
      if(lua_isstring(L,-1)){ perspRef = lua_tostring(L,-1); }
      lua_pop(L,1);
      unsigned char bgR=0,bgG=0,bgB=0,bgA=0;
      lua_getfield(L,2,"backgroundColor");
      if(lua_istable(L,-1)){
        bgR=(unsigned char)getFieldInteger(L,-1,"r",0);
        bgG=(unsigned char)getFieldInteger(L,-1,"g",0);
        bgB=(unsigned char)getFieldInteger(L,-1,"b",0);
        bgA=(unsigned char)getFieldInteger(L,-1,"a",0);
      }
      lua_pop(L,1);

      // Lighting params
      float pitch=0.f,yaw=0.f,diffusePct=60.f,diameterPct=100.f,ambientPct=30.f;
      bool rimEnabled=false;
      unsigned char lR=255,lG=255,lB=255;
      lua_getfield(L,2,"lighting");
      if(lua_istable(L,-1)){
        pitch =(float)getNum(L,-1,"pitch",0.0);
        yaw   =(float)getNum(L,-1,"yaw",0.0);
        diffusePct =(float)getNum(L,-1,"diffuse",60.0);
        diameterPct=(float)getNum(L,-1,"diameter",100.0);
        ambientPct =(float)getNum(L,-1,"ambient",30.0);
        lua_getfield(L,-1,"rimEnabled"); rimEnabled = lua_toboolean(L,-1)!=0; lua_pop(L,1);
        lua_getfield(L,-1,"lightColor");
        if(lua_istable(L,-1)){
          lR=(unsigned char)getFieldInteger(L,-1,"r",255);
          lG=(unsigned char)getFieldInteger(L,-1,"g",255);
          lB=(unsigned char)getFieldInteger(L,-1,"b",255);
        }
        lua_pop(L,1);
      }
      lua_pop(L,1);

      // Voxels
      size_t vcount=lua_rawlen(L,1);
      std::vector<Voxel> voxels;
      voxels.reserve(vcount);
      float minX=1e9f,minY=1e9f,minZ=1e9f;
      float maxX=-1e9f,maxY=-1e9f,maxZ=-1e9f;
      for(size_t i=1;i<=vcount;i++){
        lua_rawgeti(L,1,(int)i);
        if(lua_istable(L,-1)){
          int t=lua_gettop(L);
          Voxel v{0,0,0,255,255,255,255};
          lua_rawgeti(L,t,1); bool numeric=lua_isnumber(L,-1)!=0; lua_pop(L,1);
          if(numeric){
            lua_rawgeti(L,t,1); v.x=(float)lua_tonumber(L,-1); lua_pop(L,1);
            lua_rawgeti(L,t,2); v.y=(float)lua_tonumber(L,-1); lua_pop(L,1);
            lua_rawgeti(L,t,3); v.z=(float)lua_tonumber(L,-1); lua_pop(L,1);
            lua_rawgeti(L,t,4); v.r=(unsigned char)lua_tointeger(L,-1); lua_pop(L,1);
            lua_rawgeti(L,t,5); v.g=(unsigned char)lua_tointeger(L,-1); lua_pop(L,1);
            lua_rawgeti(L,t,6); v.b=(unsigned char)lua_tointeger(L,-1); lua_pop(L,1);
            lua_rawgeti(L,t,7); v.a=(unsigned char)lua_tointeger(L,-1); lua_pop(L,1);
          } else {
            v.x=(float)getNum(L,t,"x",0);
            v.y=(float)getNum(L,t,"y",0);
            v.z=(float)getNum(L,t,"z",0);
            lua_getfield(L,t,"color");
            if(lua_istable(L,-1)){
              v.r=(unsigned char)getFieldInteger(L,-1,"r",255);
              v.g=(unsigned char)getFieldInteger(L,-1,"g",255);
              v.b=(unsigned char)getFieldInteger(L,-1,"b",255);
              v.a=(unsigned char)getFieldInteger(L,-1,"a",255);
            }
            lua_pop(L,1);
          }
          if(v.x<minX)minX=v.x; if(v.x>maxX)maxX=v.x;
          if(v.y<minY)minY=v.y; if(v.y>maxY)maxY=v.y;
          if(v.z<minZ)minZ=v.z; if(v.z>maxZ)maxZ=v.z;
          voxels.push_back(v);
        }
        lua_pop(L,1);
      }

      lua_createtable(L,0,3);
      if(voxels.empty()){
        lua_pushinteger(L,width); lua_setfield(L,-2,"width");
        lua_pushinteger(L,height); lua_setfield(L,-2,"height");
        lua_pushlstring(L,"",0); lua_setfield(L,-2,"pixels");
        return 1;
      }

      // Read custom middlePoint from params if provided, otherwise calculate from bounds
      float midX, midY, midZ;
      lua_getfield(L, 2, "middlePoint");
      if (lua_istable(L, -1)) {
          midX = (float)getNum(L, -1, "x", 0.5f * (minX + maxX));
          midY = (float)getNum(L, -1, "y", 0.5f * (minY + maxY));
          midZ = (float)getNum(L, -1, "z", 0.5f * (minZ + maxZ));
      } else {
          midX = 0.5f * (minX + maxX);
          midY = 0.5f * (minY + maxY);
          midZ = 0.5f * (minZ + maxZ);
      }
      lua_pop(L, 1);
      // Use globalBounds (if provided) for camera calculations
      CameraBounds camB = computeCameraBounds(L, 2, minX, maxX, minY, maxY, minZ, maxZ, midX, midY, midZ);
      float sizeX=camB.sizeX;
      float sizeY=camB.sizeY;
      float sizeZ=camB.sizeZ;
      float maxDim=camB.maxDim;
      float effectiveSpan=camB.effectiveSpan;

      float RX=xRotDeg*(float)PI/180.f;
      float RY=yRotDeg*(float)PI/180.f;
      float RZ=zRotDeg*(float)PI/180.f;
      float cx=std::cos(RX), sx=std::sin(RX);
      float cy=std::cos(RY), sy=std::sin(RY);
      float cz=std::cos(RZ), sz=std::sin(RZ);

      // ----------------------------------------------------------------------------
      // PERSPECTIVE & ORTHOGRAPHIC CAMERA (MATCH previewRenderer.lua)
      // ----------------------------------------------------------------------------
      bool perspective = (!orth && fovDeg > 0.f);
      float focalLength = 0.f;
      float camDist;
      if (perspective) {
        fovDeg = std::max(5.f, std::min(75.f, fovDeg));
        float warpT = (fovDeg - 5.f) / (75.f - 5.f);
        if (warpT < 0.f) warpT = 0.f; if (warpT > 1.f) warpT = 1.f;
        float amplified = std::pow(warpT, 1.f/3.f);
        const float BASE_NEAR = 1.2f;
        const float FAR_EXTRA = 45.f;
        camDist = effectiveSpan * (BASE_NEAR + (1.f - amplified)*(1.f - amplified) * FAR_EXTRA);
        float fovRad = fovDeg * (float)PI / 180.f;
        focalLength = (height / 2.f) / std::tan(fovRad / 2.f);
      } else {
        camDist = effectiveSpan * 5.f;
      }

      // Compute voxelSize
      float voxelSize = scale;
      {
        float maxAllowedPixels = std::min(width, height) * viewportCutoffMultiplier;
        if (perspective && focalLength > 1e-6f && maxDim > 0.f) {
          float RX = xRotDeg*(float)PI/180.f;
          float RY = yRotDeg*(float)PI/180.f;
          float RZ = zRotDeg*(float)PI/180.f;
          float rcx=std::cos(RX), rsx=std::sin(RX);
          float rcy=std::cos(RY), rsy=std::sin(RY);
          float rcz=std::cos(RZ), rsz=std::sin(RZ);
          float zMinRot =  1e9f;
          float zMaxRot = -1e9f;
          for (int ix=0; ix<2; ++ix){
            float cxv = (ix==0)?camB.camMinX:camB.camMaxX;
            for (int iy=0; iy<2; ++iy){
              float cyv = (iy==0)?camB.camMinY:camB.camMaxY;
              for (int iz=0; iz<2; ++iz){
                float czv = (iz==0)?camB.camMinZ:camB.camMaxZ;
                float X = cxv - midX;
                float Y = cyv - midY;
                float Z = czv - midZ;
                { float Y2=Y*rcx - Z*rsx; float Z2=Y*rsx + Z*rcx; Y=Y2; Z=Z2; }
                { float X2=X*rcy + Z*rsy; float Z3=-X*rsy + Z*rcy; X=X2; Z=Z3; }
                { float X3=X*rcz - Y*rsz; float Y3=X*rsz + Y*rcz; X=X3; Y=Y3; }
                float worldZ = Z + midZ;
                if (worldZ < zMinRot) zMinRot = worldZ;
                if (worldZ > zMaxRot) zMaxRot = worldZ;
              }
            }
          }
          float cameraZ = midZ + camDist;
          float depthBack  = std::max(0.001f, cameraZ - zMinRot);
          float depthFront = std::max(0.001f, cameraZ - zMaxRot);
          float depthMiddle= std::max(0.001f, camDist);
          float depthRef = depthMiddle;
          if (perspRef == "front" || perspRef == "Front") depthRef = depthFront;
          else if (perspRef == "back" || perspRef == "Back") depthRef = depthBack;
          voxelSize = scale * (depthRef / focalLength);
          if (voxelSize * effectiveSpan > maxAllowedPixels) voxelSize = maxAllowedPixels / effectiveSpan;
          if (voxelSize <= 0.f) voxelSize = 1.f;
        } else {
          if (voxelSize < 1.f) voxelSize = 1.f;
          if (voxelSize * effectiveSpan > maxAllowedPixels && effectiveSpan > 0.f)
            voxelSize *= (maxAllowedPixels / (voxelSize * effectiveSpan));
        }
      }
      // Use slightly negative threshold to ensure edge-on faces are consistently visible
      // This prevents jagged edges at certain rotation angles where faces are nearly perpendicular
      float threshold = -0.001f;

      // Light direction
      float yawRad = yaw*(float)PI/180.f;
      float pitchRad = pitch*(float)PI/180.f;
      float cosYaw=std::cos(yawRad), sinYaw=std::sin(yawRad);
      float cosPitch=std::cos(pitchRad), sinPitch=std::sin(pitchRad);
      float Lx = cosYaw * cosPitch;
      float Ly = sinPitch;
      float Lz = sinYaw * cosPitch;
      float Lmag=std::sqrt(Lx*Lx+Ly*Ly+Lz*Lz);
      if(Lmag>1e-6f){ Lx/=Lmag; Ly/=Lmag; Lz/=Lmag; }

      auto rotateVecCam=[&](float &x,float &y,float &z){
        float y2=y*cx - z*sx; float z2=y*sx + z*cx; y=y2; z=z2;
        float x2=x*cy + z*sy; float z3=-x*sy + z*cy; x=x2; z=z3;
        float x3=x*cz - y*sz; float y3=x*sz + y*cz; x=x3; y=y3;
      };

      struct RFace { float nx,ny,nz; };
      RFace faceNormals[6];
      for(int f=0;f<6;f++){
        float nx=LOCAL_FACE_NORMALS[f][0];
        float ny=LOCAL_FACE_NORMALS[f][1];
        float nz=LOCAL_FACE_NORMALS[f][2];
        rotateVecCam(nx,ny,nz);
        float m=std::sqrt(nx*nx+ny*ny+nz*nz);
        if(m>1e-6f){ nx/=m; ny/=m; nz/=m; }
        faceNormals[f]={nx,ny,nz};
      }

      auto rotateAxis=[&](float x,float y,float z){
        rotateVecCam(x,y,z);
        return std::array<float,3>{x,y,z};
      };
      auto ex=rotateAxis(1,0,0);
      auto ey=rotateAxis(0,1,0);
      auto ez=rotateAxis(0,0,1);
      float Lmx = ex[0]*Lx + ey[0]*Ly + ez[0]*Lz;
      float Lmy = ex[1]*Lx + ey[1]*Ly + ez[1]*Lz;
      float Lmz = ex[2]*Lx + ey[2]*Ly + ez[2]*Lz;
      float Lmm=std::sqrt(Lmx*Lmx+Lmy*Lmy+Lmz*Lmz);
      if(Lmm>1e-6f){ Lmx/=Lmm; Lmy/=Lmm; Lmz/=Lmm; }

      float diag = std::sqrt(sizeX*sizeX + sizeY*sizeY + sizeZ*sizeZ);
      float modelRadius = 0.5f * diag;
      float dia = std::max(0.f, diameterPct/100.f);
      float baseRadius = dia * modelRadius;
      
      // ALIGNED WITH LUA dynamic.lua:
      // Lua: diffuseIntensity = (params.diffuse or 60) / 100
      // Lua: exponent = 1 + (1 - diffuseIntensity) * 3
      float diffNorm = diffusePct/100.f;
      float exponent = 1.f + (1.f - diffNorm) * 3.f;
      if (exponent < 0.2f) exponent = 0.2f;
      
      // ALIGNED WITH LUA: ambient is direct percentage (no 0.02 + 0.48* scaling)
      // Lua: ambientIntensity = (params.ambient or 30) / 100
      float ambient = ambientPct/100.f;
      if (ambient > 1.f) ambient = 1.f; if (ambient < 0.f) ambient = 0.f;
      
      float lightCr = (float)lR / 255.f;
      float lightCg = (float)lG / 255.f;
      float lightCb = (float)lB / 255.f;

      struct Poly { 
        float x[4],y[4],depth; 
        unsigned char r,g,b,a;
        // Tiebreaker fields for stable sorting
        float voxelX, voxelY, voxelZ;
        int faceIdx;
      };
      std::vector<Poly> polys;
      polys.reserve(voxels.size()*6);
      float screenCX=width*0.5f;
      float screenCY=height*0.5f;
      float camZ = midZ + camDist;

      for(auto &v: voxels){
        float x=v.x-midX, y=v.y-midY, z=v.z-midZ;
        { float y2=y*cx - z*sx; float z2=y*sx + z*cx; y=y2; z=z2; }
        { float x2=x*cy + z*sy; double z3=-x*sy + z*cy; x=x2; z=(float)z3; }
        { float x3=x*cz - y*sz; float y3=x*sz + y*cz; x=x3; y=y3; }
        float worldX=x+midX, worldY=y+midY, worldZ=z+midZ;

        float vxv,vyv,vzv;
        if(orth){ vxv=0.f; vyv=0.f; vzv=1.f; }
        else {
          vxv=midX-worldX; vyv=midY-worldY; vzv=camZ-worldZ;
          float m=std::sqrt(vxv*vxv+vyv*vyv+vzv*vzv);
          if(m>1e-5f){ vxv/=m; vyv/=m; vzv/=m; }
        }

        // ALIGNED WITH LUA dynamic.lua radial attenuation:
        // Lua: perpDist = perpendicular distance from light axis
        // Lua: radialFactor = max(0, 1 - perpDist / radius)
        float mvx=v.x-midX, mvy=v.y-midY, mvz=v.z-midZ;
        float proj = mvx*Lmx + mvy*Lmy + mvz*Lmz;  // Project onto light dir
        float px = mvx - proj*Lmx;  // Perpendicular component
        float py = mvy - proj*Lmy;
        float pz = mvz - proj*Lmz;
        float perp=std::sqrt(px*px+py*py+pz*pz);
        float radial=1.f;
        if(baseRadius>1e-6f){
          // Simple linear falloff matching Lua
          radial = 1.f - (perp / baseRadius);
          if(radial < 0.f) radial = 0.f;
        }

        for(int f=0;f<6;f++){
          const auto &fn=faceNormals[f];
          float visDot=fn.nx*vxv + fn.ny*vyv + fn.nz*vzv;
          if(visDot <= threshold) continue;
          float ndotl = fn.nx*Lx + fn.ny*Ly + fn.nz*Lz;
          if(ndotl < 0.f) ndotl=0.f;
          // ALIGNED WITH LUA: diffuse = ndotl^exp * radial * diffuseIntensity
          float diff = std::pow(ndotl, exponent) * radial * diffNorm;
          float rF=v.r * (ambient + diff * lightCr);
          float gF=v.g * (ambient + diff * lightCg);
          float bF=v.b * (ambient + diff * lightCb);
          if(rimEnabled){
            float ndotv=fn.nz; // viewDir (0,0,1)
            if(ndotv > 0.f){
              float edge=1.f - ndotv;
              float t;
              if(edge <= 0.55f) t=0.f; else if(edge >= 0.95f) t=1.f;
              else { float tt=(edge-0.55f)/(0.4f); t=tt*tt*(3.f-2.f*tt); }
              if(t>0.f){
                float rimStrength=0.6f;
                float rim=rimStrength*t;
                rF += lightCr * rim * 255.f;
                gF += lightCg * rim * 255.f;
                bF += rimStrength * rim * 255.f;
              }
            }
          }
          if(rF<0) rF=0; if(rF>255) rF=255;
          if(gF<0) gF=0; if(gF>255) gF=255;
          if(bF<0) bF=0; if(bF>255) bF=255;

          Poly poly{};
          poly.r=(unsigned char)std::lround(rF);
          poly.g=(unsigned char)std::lround(gF);
          poly.b=(unsigned char)std::lround(bF);
          poly.a=v.a;
          // Store tiebreaker info for stable sorting
          poly.voxelX = v.x;
          poly.voxelY = v.y;
          poly.voxelZ = v.z;
          poly.faceIdx = f;
          float avgDepth=0.f;
          for(int vi=0;vi<4;vi++){
            int vid=FACE_IDX[f][vi]-1;
            float lx=UNIT_VERTS[vid][0]*voxelSize;
            float ly=UNIT_VERTS[vid][1]*voxelSize;
            float lz=UNIT_VERTS[vid][2]*voxelSize;
            { float y2=ly*cx - lz*sx; float z2=ly*sx + lz*cx; ly=y2; lz=z2; }
            { float x2=lx*cy + lz*sy; float z3=-lx*sy + lz*cy; lx=x2; lz=z3; }
            { float x3=lx*cz - ly*sz; float y3=lx*sz + ly*cz; lx=x3; ly=y3; }
            float wx=worldX*voxelSize + lx;
            float wy=worldY*voxelSize + ly;
            float wz=worldZ + (lz/voxelSize);
            if(perspective){
              float depth=(camZ - wz);
              if(depth<0.001f) depth=0.001f;
              float s=focalLength>0? (focalLength/depth):1.f;
              poly.x[vi]=screenCX + (wx - midX*voxelSize)*s;
              poly.y[vi]=screenCY + (wy - midY*voxelSize)*s;
              avgDepth += depth;
            } else {
              poly.x[vi]=screenCX + (wx - midX*voxelSize);
              poly.y[vi]=screenCY + (wy - midY*voxelSize);
              avgDepth += (camZ - worldZ);
            }
          }
          poly.depth=avgDepth/4.f;
          polys.push_back(poly);
        }
      }

      // --- Preview voxels (ghost style, no lighting) ---
      {
        auto previewVoxels = readPreviewVoxels(L, 2);
        for (auto& pv : previewVoxels) {
          float x = pv.x - midX, y = pv.y - midY, z = pv.z - midZ;
          { float y2 = y*cx - z*sx; float z2 = y*sx + z*cx; y = y2; z = z2; }
          { float x2 = x*cy + z*sy; float z3 = -x*sy + z*cy; x = x2; z = z3; }
          { float x3 = x*cz - y*sz; float y3 = x*sz + y*cz; x = x3; y = y3; }
          float worldX = x+midX, worldY = y+midY, worldZ = z+midZ;

          float vxv, vyv, vzv;
          if(orth){ vxv=0; vyv=0; vzv=1; }
          else {
            vxv=midX-worldX; vyv=midY-worldY; vzv=camZ-worldZ;
            float m=std::sqrt(vxv*vxv+vyv*vyv+vzv*vzv);
            if(m>1e-5f){vxv/=m;vyv/=m;vzv/=m;}
          }

          for(int f=0;f<6;f++){
            float fnx=LOCAL_FACE_NORMALS[f][0], fny=LOCAL_FACE_NORMALS[f][1], fnz=LOCAL_FACE_NORMALS[f][2];
            { float y2=fny*cx-fnz*sx; float z2=fny*sx+fnz*cx; fny=y2; fnz=z2; }
            { float x2=fnx*cy+fnz*sy; float z3=-fnx*sy+fnz*cy; fnx=x2; fnz=z3; }
            { float x3=fnx*cz-fny*sz; float y3=fnx*sz+fny*cz; fnx=x3; fny=y3; }
            float dot=fnx*vxv+fny*vyv+fnz*vzv;
            if(dot<=threshold) continue;

            Poly poly;
            poly.r=pv.r; poly.g=pv.g; poly.b=pv.b; poly.a=pv.a;
            poly.voxelX=pv.x; poly.voxelY=pv.y; poly.voxelZ=pv.z;
            poly.faceIdx=f;
            float avgDepth=0.f;
            for(int vi=0;vi<4;vi++){
              int vid=FACE_IDX[f][vi]-1;
              float lx=UNIT_VERTS[vid][0]*voxelSize;
              float ly=UNIT_VERTS[vid][1]*voxelSize;
              float lz=UNIT_VERTS[vid][2]*voxelSize;
              { float y2=ly*cx-lz*sx; float z2=ly*sx+lz*cx; ly=y2; lz=z2; }
              { float x2=lx*cy+lz*sy; float z3=-lx*sy+lz*cy; lx=x2; lz=z3; }
              { float x3=lx*cz-ly*sz; float y3=lx*sz+ly*cz; lx=x3; ly=y3; }
              float wx=worldX*voxelSize+lx;
              float wy=worldY*voxelSize+ly;
              float wz=worldZ+(lz/voxelSize);
              if(perspective){
                float depth=std::max(0.001f,camZ-wz);
                float s=focalLength>0?(focalLength/depth):1.f;
                poly.x[vi]=screenCX+(wx-midX*voxelSize)*s;
                poly.y[vi]=screenCY+(wy-midY*voxelSize)*s;
                avgDepth+=depth;
              } else {
                poly.x[vi]=screenCX+(wx-midX*voxelSize);
                poly.y[vi]=screenCY+(wy-midY*voxelSize);
                avgDepth+=(camZ-worldZ);
              }
            }
            poly.depth=avgDepth/4.f;
            polys.push_back(poly);
          }
        }
      }

      // Stable sort: primary by depth (far to near), secondary by voxel position + face for tiebreaker
      std::sort(polys.begin(),polys.end(),[](const Poly&a,const Poly&b){
        constexpr float DEPTH_EPS = 0.0001f;
        if (std::abs(a.depth - b.depth) > DEPTH_EPS) {
          return a.depth > b.depth;
        }
        // Tiebreaker: consistent ordering by voxel position, then face index
        if (a.voxelZ != b.voxelZ) return a.voxelZ < b.voxelZ;
        if (a.voxelY != b.voxelY) return a.voxelY < b.voxelY;
        if (a.voxelX != b.voxelX) return a.voxelX < b.voxelX;
        return a.faceIdx < b.faceIdx;
      });

      std::vector<unsigned char> buffer((size_t)width*height*4,0);
      for(int y=0;y<height;y++){
        for(int x=0;x<width;x++){
          size_t off=((size_t)y*width + x)*4;
          buffer[off+0]=bgR; buffer[off+1]=bgG; buffer[off+2]=bgB; buffer[off+3]=bgA;
        }
      }
      for(auto &p: polys){
        FacePoly polyOut;
        for(int i=0;i<4;i++){ polyOut.x[i]=p.x[i]; polyOut.y[i]=p.y[i]; }
        polyOut.depth=p.depth; polyOut.r=p.r; polyOut.g=p.g; polyOut.b=p.b; polyOut.a=p.a;
        rasterQuad(polyOut,width,height,buffer);
      }
      lua_pushinteger(L,width); lua_setfield(L,-2,"width");
      lua_pushinteger(L,height); lua_setfield(L,-2,"height");
      lua_pushlstring(L,(const char*)buffer.data(), buffer.size()); lua_setfield(L,-2,"pixels");
      return 1;
  }},
  {nullptr,nullptr}
};

#ifdef _WIN32
  #define ASEVOXEL_API __declspec(dllexport)
  #include <windows.h>
#else
  #define ASEVOXEL_API
  #include <dlfcn.h>
#endif

// ---------------------------------------------------------------------------
// Schedules FreeLibrary on this DLL to run on a background thread AFTER the
// current Lua call has fully returned.  This is the only safe way to call
// FreeLibrary on the DLL that is currently executing:
//
//   - Calling FreeLibrary directly from inside the DLL unmaps it while the
//     return address is still pointing into it -> instant crash.
//   - CreateThread defers the FreeLibrary call until the thread scheduler
//     runs, which is after the call stack has fully unwound back through Lua
//     and the DLL's code is no longer on any thread's stack.
//   - The thread entry point is a simple WINAPI thunk; it only executes
//     FreeLibrary(hSelf) and exits.  By the time the thread starts, Lua has
//     already returned from this function and the DLL call frame is gone.
//
// On Linux/macOS this function does nothing (file locks are not held on
// loaded .so files; the file can be deleted while mapped).
// ---------------------------------------------------------------------------
// asevoxel_schedule_unload()
//
// Windows strategy: rename the DLL to a random temp name in the same
// directory while it is still loaded.  Windows allows renaming (not
// deleting) open files via MoveFileExW.  After renaming, the directory
// entry for the original filename is gone, so Aseprite can delete the
// extension folder.  The renamed file stays mapped until the process exits,
// at which point Windows cleans it up automatically.
//
// This avoids all threading/timing hazards: no background thread, no
// busy-wait, no FreeLibrary-while-executing race.
extern "C" ASEVOXEL_API int asevoxel_schedule_unload(lua_State* L) {
#ifdef _WIN32
  // Get the full path of this DLL.
  WCHAR dllPath[MAX_PATH] = {0};
  HMODULE hSelf = NULL;
  GetModuleHandleExW(
    GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
    GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
    (LPCWSTR)(void*)&asevoxel_schedule_unload,
    &hSelf);
  if (!hSelf || !GetModuleFileNameW(hSelf, dllPath, MAX_PATH)) {
    lua_pushboolean(L, 0);
    return 1;
  }

  // Build a temp path: same directory, name = "_asevoxel_unloading_<tick>.dll"
  WCHAR tempPath[MAX_PATH] = {0};
  WCHAR* lastSlash = wcsrchr(dllPath, L'\\');
  if (!lastSlash) {
    lua_pushboolean(L, 0);
    return 1;
  }
  size_t dirLen = (size_t)(lastSlash - dllPath) + 1;
  wcsncpy_s(tempPath, MAX_PATH, dllPath, dirLen);
  // Append a unique temp name
  DWORD tick = GetTickCount();
  wsprintfW(tempPath + dirLen, L"_asevoxel_unloading_%u.dll", tick);

  // Rename: this removes the original directory entry so the folder can be
  // deleted.  The file data stays mapped; it will be deleted by Windows when
  // the last handle is closed (at process exit).
  BOOL moved = MoveFileExW(dllPath, tempPath, 0);

  // Push: true = renamed ok, false = rename failed (non-fatal; Aseprite may
  // still fail to delete, but we won't crash).
  lua_pushboolean(L, moved ? 1 : 0);
  if (!moved) {
    // Push error code as second return so Lua can print it
    lua_pushinteger(L, (lua_Integer)GetLastError());
    return 2;
  }
#else
  lua_pushboolean(L, 0);
#endif
  return 1;
}

// ---------------------------------------------------------------------------
// BUG-009: native process spawning that never requests a console.
//
// compute_bridge.lua's run_binary()/isAvailable() and daemon_bridge.lua's
// launch() all used to shell out via os.execute()/io.popen(), which on
// Windows always goes through cmd.exe. Aseprite.exe is a console-less GUI
// process, so every such call needs Windows to create a *new* console for
// the hidden cmd.exe. When Windows Terminal is the system's default terminal
// application (the modern Windows 11 default) and no WindowsTerminal.exe
// process happens to already be running, that console-creation handoff has
// to cold-start a whole new WindowsTerminal.exe process first -- a
// documented, upstream-confirmed delay of 5+ seconds per occurrence (see
// microsoft/terminal#19602, #11719). This exactly matches BUG-009's ~5.5-6s
// per-switch stall, reproduced directly and root-caused in that ticket.
//
// CreateProcess with DETACHED_PROCESS requests *no console at all* -- there
// is nothing to hand off, so this bypasses the whole mechanism regardless of
// the end user's Windows Terminal / default-terminal-application settings
// (which AseVoxel has no way to change on their machine, and shouldn't need
// to -- see BUG-009's "Eighth session" for the system-setting workaround
// that isn't a real fix for other users).
// ---------------------------------------------------------------------------

// Quotes a single argument per the MS C runtime command-line parsing rules
// (https://learn.microsoft.com/en-us/cpp/c-language/parsing-c-command-line-arguments),
// so CreateProcess's single command-line string round-trips correctly
// through the child's own argv parsing.
static std::string quote_arg(const std::string& arg) {
  if (!arg.empty() && arg.find_first_of(" \t\n\v\"") == std::string::npos) {
    return arg;
  }
  std::string out = "\"";
  for (size_t i = 0; i < arg.size(); ) {
    size_t backslashes = 0;
    while (i < arg.size() && arg[i] == '\\') { backslashes++; i++; }
    if (i == arg.size()) {
      out.append(backslashes * 2, '\\');
      break;
    } else if (arg[i] == '"') {
      out.append(backslashes * 2 + 1, '\\');
      out += '"';
      i++;
    } else {
      out.append(backslashes, '\\');
      out += arg[i];
      i++;
    }
  }
  out += '"';
  return out;
}

static std::string build_command_line(const std::string& exePath, const std::vector<std::string>& args) {
  std::string cmd = quote_arg(exePath);
  for (const auto& a : args) {
    cmd += ' ';
    cmd += quote_arg(a);
  }
  return cmd;
}

static std::vector<std::string> lua_string_array(lua_State* L, int idx) {
  std::vector<std::string> out;
#ifdef _WIN32
  lua_Integer n = (lua_Integer)lua_rawlen(L, idx);
  for (lua_Integer i = 1; i <= n; i++) {
    lua_geti(L, idx, i);
    const char* s = lua_tostring(L, -1);
    out.push_back(s ? s : "");
    lua_pop(L, 1);
  }
#endif
  return out;
}

#ifdef _WIN32
static void read_pipe_to_string(HANDLE h, std::string* out) {
  char buf[8192];
  DWORD n = 0;
  while (ReadFile(h, buf, sizeof(buf), &n, NULL) && n > 0) {
    out->append(buf, n);
  }
}
#endif

// spawn_capture(exePath, argsArray) -> stdoutData, stderrData, exitCode
//   or: nil, errorMessage
//
// Blocking: waits for the child to exit, capturing all of stdout and
// stderr. No stdin is provided (nothing in this codebase's usage needs it --
// input is passed via --stdin-file, a real temp file, not piped stdin).
extern "C" ASEVOXEL_API int asevoxel_spawn_capture(lua_State* L) {
  const char* exePath = luaL_checkstring(L, 1);
  luaL_checktype(L, 2, LUA_TTABLE);
#ifdef _WIN32
  std::vector<std::string> args = lua_string_array(L, 2);
  std::string cmdLine = build_command_line(exePath, args);

  SECURITY_ATTRIBUTES sa;
  sa.nLength = sizeof(sa);
  sa.bInheritHandle = TRUE;
  sa.lpSecurityDescriptor = NULL;

  HANDLE stdoutRead = NULL, stdoutWrite = NULL;
  HANDLE stderrRead = NULL, stderrWrite = NULL;
  if (!CreatePipe(&stdoutRead, &stdoutWrite, &sa, 1 << 20) ||
      !CreatePipe(&stderrRead, &stderrWrite, &sa, 1 << 16)) {
    lua_pushnil(L);
    lua_pushstring(L, "CreatePipe failed");
    return 2;
  }
  SetHandleInformation(stdoutRead, HANDLE_FLAG_INHERIT, 0);
  SetHandleInformation(stderrRead, HANDLE_FLAG_INHERIT, 0);

  HANDLE nulRead = CreateFileA("NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                &sa, OPEN_EXISTING, 0, NULL);

  STARTUPINFOA si;
  ZeroMemory(&si, sizeof(si));
  si.cb = sizeof(si);
  si.dwFlags = STARTF_USESTDHANDLES;
  si.hStdInput  = nulRead;
  si.hStdOutput = stdoutWrite;
  si.hStdError  = stderrWrite;

  PROCESS_INFORMATION pi;
  ZeroMemory(&pi, sizeof(pi));

  std::vector<char> cmdLineBuf(cmdLine.begin(), cmdLine.end());
  cmdLineBuf.push_back('\0');

  BOOL ok = CreateProcessA(
      NULL, cmdLineBuf.data(), NULL, NULL, TRUE,
      DETACHED_PROCESS, NULL, NULL, &si, &pi);

  CloseHandle(stdoutWrite);
  CloseHandle(stderrWrite);
  if (nulRead != INVALID_HANDLE_VALUE) CloseHandle(nulRead);

  if (!ok) {
    DWORD err = GetLastError();
    CloseHandle(stdoutRead);
    CloseHandle(stderrRead);
    lua_pushnil(L);
    lua_pushfstring(L, "CreateProcess failed, error %d", (int)err);
    return 2;
  }
  CloseHandle(pi.hThread);

  std::string stdoutData, stderrData;
  // stderr read on a background thread: it only ever carries small JSON
  // diagnostics in this codebase's usage, but reading it concurrently with
  // stdout (which carries the large RGBA payload) rules out any deadlock if
  // both pipes filled at once.
  std::thread stderrThread([&]() { read_pipe_to_string(stderrRead, &stderrData); });
  read_pipe_to_string(stdoutRead, &stdoutData);
  stderrThread.join();

  CloseHandle(stdoutRead);
  CloseHandle(stderrRead);

  WaitForSingleObject(pi.hProcess, INFINITE);
  DWORD exitCode = 0;
  GetExitCodeProcess(pi.hProcess, &exitCode);
  CloseHandle(pi.hProcess);

  lua_pushlstring(L, stdoutData.data(), stdoutData.size());
  lua_pushlstring(L, stderrData.data(), stderrData.size());
  lua_pushinteger(L, (lua_Integer)exitCode);
  return 3;
#else
  lua_pushnil(L);
  lua_pushstring(L, "spawn_capture not implemented on this platform (os.execute/io.popen have no console-handoff issue outside Windows)");
  return 2;
#endif
}

// spawn_detached(exePath, argsArray) -> pid
//   or: nil, errorMessage
//
// Fire-and-forget: does not wait for exit or capture output. Used for the
// long-running Compute Daemon process; reachability is checked separately
// over its WebSocket port, not by reading its stdio.
extern "C" ASEVOXEL_API int asevoxel_spawn_detached(lua_State* L) {
  const char* exePath = luaL_checkstring(L, 1);
  luaL_checktype(L, 2, LUA_TTABLE);
#ifdef _WIN32
  std::vector<std::string> args = lua_string_array(L, 2);
  std::string cmdLine = build_command_line(exePath, args);

  SECURITY_ATTRIBUTES sa;
  sa.nLength = sizeof(sa);
  sa.bInheritHandle = TRUE;
  sa.lpSecurityDescriptor = NULL;

  HANDLE nulRead  = CreateFileA("NUL", GENERIC_READ,  FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, OPEN_EXISTING, 0, NULL);
  HANDLE nulWrite = CreateFileA("NUL", GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, OPEN_EXISTING, 0, NULL);

  STARTUPINFOA si;
  ZeroMemory(&si, sizeof(si));
  si.cb = sizeof(si);
  si.dwFlags = STARTF_USESTDHANDLES;
  si.hStdInput  = nulRead;
  si.hStdOutput = nulWrite;
  si.hStdError  = nulWrite;

  PROCESS_INFORMATION pi;
  ZeroMemory(&pi, sizeof(pi));

  std::vector<char> cmdLineBuf(cmdLine.begin(), cmdLine.end());
  cmdLineBuf.push_back('\0');

  BOOL ok = CreateProcessA(
      NULL, cmdLineBuf.data(), NULL, NULL, TRUE,
      DETACHED_PROCESS, NULL, NULL, &si, &pi);

  if (nulRead  != INVALID_HANDLE_VALUE) CloseHandle(nulRead);
  if (nulWrite != INVALID_HANDLE_VALUE) CloseHandle(nulWrite);

  if (!ok) {
    DWORD err = GetLastError();
    lua_pushnil(L);
    lua_pushfstring(L, "CreateProcess failed, error %d", (int)err);
    return 2;
  }

  DWORD pid = pi.dwProcessId;
  CloseHandle(pi.hThread);
  CloseHandle(pi.hProcess);

  lua_pushinteger(L, (lua_Integer)pid);
  return 1;
#else
  lua_pushnil(L);
  lua_pushstring(L, "spawn_detached not implemented on this platform");
  return 2;
#endif
}

extern "C" ASEVOXEL_API int luaopen_asevoxel_native(lua_State* L) {
#if LUA_VERSION_NUM >= 502
  luaL_newlib(L, FUNCS);
#else
  lua_newtable(L);
  for (const luaL_Reg* r=FUNCS; r->name; ++r) {
    lua_pushcfunction(L,r->func);
    lua_setfield(L,-2,r->name);
  }
#endif
  lua_pushstring(L,"0.1.0"); lua_setfield(L,-2,"version");
  lua_pushstring(L,"asevoxel_native"); lua_setfield(L,-2,"name");
  return 1;
}