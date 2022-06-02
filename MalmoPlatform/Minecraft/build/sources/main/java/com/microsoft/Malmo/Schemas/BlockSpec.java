//
// This file was generated by the JavaTM Architecture for XML Binding(JAXB) Reference Implementation, v2.2.4 
// See <a href="http://java.sun.com/xml/jaxb">http://java.sun.com/xml/jaxb</a> 
// Any modifications to this file will be lost upon recompilation of the source schema. 
// Generated on: 2022.06.01 at 02:32:08 PM EEST 
//


package com.microsoft.Malmo.Schemas;

import java.util.ArrayList;
import java.util.List;
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlSeeAlso;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for BlockSpec complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="BlockSpec">
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;attribute name="type" use="required" type="{http://ProjectMalmo.microsoft.com}BlockList" />
 *       &lt;attribute name="variant" type="{http://ProjectMalmo.microsoft.com}VariantList" />
 *       &lt;attribute name="colour" type="{http://ProjectMalmo.microsoft.com}ColourList" />
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "BlockSpec")
@XmlSeeAlso({
    BlockSpecWithRewardAndBehaviour.class,
    BlockSpecWithDescription.class
})
public class BlockSpec {

    @XmlAttribute(name = "type", required = true)
    protected List<BlockType> type;
    @XmlAttribute(name = "variant")
    protected List<Variation> variant;
    @XmlAttribute(name = "colour")
    protected List<Colour> colour;

    /**
     * Gets the value of the type property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the type property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getType().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link BlockType }
     * 
     * 
     */
    public List<BlockType> getType() {
        if (type == null) {
            type = new ArrayList<BlockType>();
        }
        return this.type;
    }

    /**
     * Gets the value of the variant property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the variant property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getVariant().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link Variation }
     * 
     * 
     */
    public List<Variation> getVariant() {
        if (variant == null) {
            variant = new ArrayList<Variation>();
        }
        return this.variant;
    }

    /**
     * Gets the value of the colour property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the colour property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getColour().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link Colour }
     * 
     * 
     */
    public List<Colour> getColour() {
        if (colour == null) {
            colour = new ArrayList<Colour>();
        }
        return this.colour;
    }

}
