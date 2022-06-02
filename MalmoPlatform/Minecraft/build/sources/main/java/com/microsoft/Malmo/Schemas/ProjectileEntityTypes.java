//
// This file was generated by the JavaTM Architecture for XML Binding(JAXB) Reference Implementation, v2.2.4 
// See <a href="http://java.sun.com/xml/jaxb">http://java.sun.com/xml/jaxb</a> 
// Any modifications to this file will be lost upon recompilation of the source schema. 
// Generated on: 2022.06.01 at 02:32:08 PM EEST 
//


package com.microsoft.Malmo.Schemas;

import javax.xml.bind.annotation.XmlEnum;
import javax.xml.bind.annotation.XmlEnumValue;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for ProjectileEntityTypes.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="ProjectileEntityTypes">
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string">
 *     &lt;enumeration value="XPOrb"/>
 *     &lt;enumeration value="AreaEffectCloud"/>
 *     &lt;enumeration value="ThrownEgg"/>
 *     &lt;enumeration value="Arrow"/>
 *     &lt;enumeration value="Snowball"/>
 *     &lt;enumeration value="Fireball"/>
 *     &lt;enumeration value="SmallFireball"/>
 *     &lt;enumeration value="ThrownEnderpearl"/>
 *     &lt;enumeration value="EyeOfEnderSignal"/>
 *     &lt;enumeration value="ThrownPotion"/>
 *     &lt;enumeration value="ThrownExpBottle"/>
 *     &lt;enumeration value="WitherSkull"/>
 *     &lt;enumeration value="FireworksRocketEntity"/>
 *     &lt;enumeration value="SpectralArrow"/>
 *     &lt;enumeration value="ShulkerBullet"/>
 *     &lt;enumeration value="DragonFireball"/>
 *     &lt;enumeration value="EvocationFangs"/>
 *     &lt;enumeration value="LlamaSpit"/>
 *   &lt;/restriction>
 * &lt;/simpleType>
 * </pre>
 * 
 */
@XmlType(name = "ProjectileEntityTypes")
@XmlEnum
public enum ProjectileEntityTypes {

    @XmlEnumValue("XPOrb")
    XP_ORB("XPOrb"),
    @XmlEnumValue("AreaEffectCloud")
    AREA_EFFECT_CLOUD("AreaEffectCloud"),
    @XmlEnumValue("ThrownEgg")
    THROWN_EGG("ThrownEgg"),
    @XmlEnumValue("Arrow")
    ARROW("Arrow"),
    @XmlEnumValue("Snowball")
    SNOWBALL("Snowball"),
    @XmlEnumValue("Fireball")
    FIREBALL("Fireball"),
    @XmlEnumValue("SmallFireball")
    SMALL_FIREBALL("SmallFireball"),
    @XmlEnumValue("ThrownEnderpearl")
    THROWN_ENDERPEARL("ThrownEnderpearl"),
    @XmlEnumValue("EyeOfEnderSignal")
    EYE_OF_ENDER_SIGNAL("EyeOfEnderSignal"),
    @XmlEnumValue("ThrownPotion")
    THROWN_POTION("ThrownPotion"),
    @XmlEnumValue("ThrownExpBottle")
    THROWN_EXP_BOTTLE("ThrownExpBottle"),
    @XmlEnumValue("WitherSkull")
    WITHER_SKULL("WitherSkull"),
    @XmlEnumValue("FireworksRocketEntity")
    FIREWORKS_ROCKET_ENTITY("FireworksRocketEntity"),
    @XmlEnumValue("SpectralArrow")
    SPECTRAL_ARROW("SpectralArrow"),
    @XmlEnumValue("ShulkerBullet")
    SHULKER_BULLET("ShulkerBullet"),
    @XmlEnumValue("DragonFireball")
    DRAGON_FIREBALL("DragonFireball"),
    @XmlEnumValue("EvocationFangs")
    EVOCATION_FANGS("EvocationFangs"),
    @XmlEnumValue("LlamaSpit")
    LLAMA_SPIT("LlamaSpit");
    private final String value;

    ProjectileEntityTypes(String v) {
        value = v;
    }

    public String value() {
        return value;
    }

    public static ProjectileEntityTypes fromValue(String v) {
        for (ProjectileEntityTypes c: ProjectileEntityTypes.values()) {
            if (c.value.equals(v)) {
                return c;
            }
        }
        throw new IllegalArgumentException(v);
    }

}
