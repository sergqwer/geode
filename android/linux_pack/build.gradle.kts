// The agent's Linux image (tools/linux: build.sh, slim.sh, pack.sh) as an install-time asset pack: Google Play
// delivers it with the app (native code reaches a Play app through Play only), Geode unpacks it once (Agent.install).
plugins {
    id("com.android.asset-pack")
}

assetPack {
    packName.set("linux")
    dynamicDelivery {
        deliveryType.set("install-time")
    }
}
