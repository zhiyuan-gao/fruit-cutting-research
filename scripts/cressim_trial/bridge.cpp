// Headless interface to the public CRESSim-MPM scene API. Solver units: cm, g, s.
#include "simulation_factory.h"
#include <algorithm>
#include <cmath>

using namespace crmpm;

struct Trial {
    SimulationFactory *factory;
    Scene *scene;
    Shape *knife;
    int n;
    float plane;
    static constexpr float bladeHalfHeight = 3.07f;
};

static Shape *addShape(Trial &t, GeometryType geometry, float4 dimensions,
                       Vec3f center, ShapeType kind, float fatten, float friction) {
    auto *g = t.factory->createGeometry(geometry, dimensions);
    ShapeDesc s{};
    s.type = kind;
    s.transform.position = center;
    s.sdfFatten = fatten;
    s.sdfSmoothDistance = 0;
    s.friction = friction;
    s.drag = 0;
    auto *shape = t.factory->createShape(*g, s);
    t.scene->addShape(shape);
    g->release();
    return shape;
}

extern "C" Trial *start_trial(int n, const float *xyzMass, const float *elastic,
                             float plane, float tip, float dt, float dx, float fatten) {
    auto *t = new Trial{};
    t->n = n;
    t->plane = plane;
    t->factory = createFactory(n, 4, 4, 64, true);
    SceneDesc sd{};
    sd.solverType = MpmSolverType::eGpuMlsMpmSolver;
    sd.numMaxParticles = n;
    sd.gravity = Vec3f(0, 0, -981);
    sd.gridBounds = Bounds3(Vec3f(-12,-30,-3), Vec3f(8,30,35));
    sd.gridCellSize = dx;
    sd.solverIntegrationStepSize = dt;
    t->scene = t->factory->createScene(sd);
    unsigned offset;
    if (!t->scene->allocateParticles(n, offset) || offset != 0) return nullptr;
    auto &p = t->scene->getParticleData();
    auto &m = t->scene->getParticleMaterialData();
    for (int i=0; i<n; ++i) {
        const float *v = xyzMass + 4*i;
        p.positionMass[i] = make_float4(v[0],v[1],v[2],v[3]);
        p.velocity[i] = Vec3f(0);
        m.params0[i] = make_float4(elastic[2*i],elastic[2*i+1],0,0);
        m.type[i] = ParticleMaterialType::eCoRotational;
        t->scene->getComputeInitialDataIndices().pushBack(i);
    }
    t->scene->setActiveMaskRange(0,n,1);
    t->scene->markDirty(SceneDataDirtyFlags::eAll);
    for (int side : {-1,1}) {
        const float lo = side<0 ? -18 : plane+1.2f;
        const float hi = side<0 ? plane-1.2f : 18;
        addShape(*t,GeometryType::eBox,make_float4(15,(hi-lo)/2,1.3f,0),
                 Vec3f(0,(hi+lo)/2,1.8f),ShapeType::eStatic,0,.35f)->release();
    }
    t->knife = addShape(*t,GeometryType::eQuadSlicer,make_float4(8,Trial::bladeHalfHeight,0,0),
                        Vec3f(-2.5f,plane,tip+Trial::bladeHalfHeight),ShapeType::eKinematic,fatten,0);
    return t;
}

extern "C" void advance_trial(Trial *t, float tip, float duration) {
    Transform pose;
    pose.position = Vec3f(-2.5f,t->plane,tip+Trial::bladeHalfHeight);
    t->knife->setKinematicTarget(pose,duration);
    t->scene->advance(duration);
    t->scene->fetchResults();
}

extern "C" void positions(Trial *t, float *dst) {
    const auto *p=t->scene->getParticleData().positionMass;
    for (int i=0; i<t->n; ++i) {
        dst[3*i]=.01f*p[i].x; dst[3*i+1]=.01f*p[i].y; dst[3*i+2]=.01f*p[i].z;
    }
}

extern "C" void add_y_velocity(Trial *t, const float *delta) {
    auto *v=t->scene->getParticleData().velocity;
    for (int i=0; i<t->n; ++i) v[i].y() += delta[i]*100;
    t->scene->markDirty(SceneDataDirtyFlags::eParticleVelocity);
}

// A prescribed axial displacement of small pull patches, with lateral and
// vertical motion left free. A matched hold branch prescribes zero displacement.
extern "C" void drive_grips(Trial *t, const signed char *sign, const float *startY,
                            float displacement, float speed) {
    auto &p=t->scene->getParticleData();
    for (int i=0; i<t->n; ++i) {
        if (!sign[i]) continue;
        p.positionMass[i].y = (startY[i] + sign[i]*displacement)*100;
        p.velocity[i].y() = sign[i]*speed*100;
    }
    t->scene->markDirty(SceneDataDirtyFlags::eParticlePositionMass);
    t->scene->markDirty(SceneDataDirtyFlags::eParticleVelocity);
}

// Used only in the explicit post-cut branch. Keep all solver deformation state
// of the retained material and turn off the complementary particles on its grid.
extern "C" void retain_material(Trial *t, const unsigned char *keep) {
    int i=0;
    while (i<t->n) {
        if (keep[i]) { ++i; continue; }
        int j=i+1;
        while (j<t->n && !keep[j]) ++j;
        t->scene->setActiveMaskRange(i,j-i,0);
        i=j;
    }
}

extern "C" void finish_trial(Trial *t) {
    t->knife->release(); t->scene->release(); releaseFactory(t->factory); delete t;
}
